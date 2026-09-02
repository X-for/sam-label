from __future__ import annotations

import asyncio

from .exporters import export_result
from .model_runtime import Sam3Runtime
from .schemas import JobStatus
from .store import JobStore


class JobWorker:
    def __init__(self, store: JobStore, runtime: Sam3Runtime):
        self.store = store
        self.runtime = runtime
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def enqueue(self, job_id: str) -> None:
        self.runtime.queue_became_active()
        self.queue.put_nowait(job_id)

    async def run(self) -> None:
        while True:
            job_id = await self.queue.get()
            try:
                if job_id is None:
                    return
                await self._process(job_id)
            finally:
                self.queue.task_done()
                if job_id is not None and self.queue.empty():
                    self.runtime.queue_became_idle()

    async def _process(self, job_id: str) -> None:
        job = await self.store.get(job_id)
        if job is None:
            return
        await self.store.set_status(job_id, JobStatus.RUNNING)
        try:
            loop = asyncio.get_running_loop()

            def report_progress(processed_images: int, total_images: int) -> None:
                if total_images != len(job.images):
                    raise ValueError("runtime progress total does not match job image count")
                future = asyncio.run_coroutine_threadsafe(
                    self.store.set_progress(job_id, processed_images), loop
                )
                future.result()

            result = await self.runtime.predict(job, progress_callback=report_progress)
            result_dir = self.store.result_dir / job_id
            path = await asyncio.to_thread(export_result, result, job.config, result_dir)
            await self.store.set_status(job_id, JobStatus.SUCCEEDED, result_path=str(path))
        except Exception as exc:
            await self.store.set_status(job_id, JobStatus.FAILED, error=str(exc))
