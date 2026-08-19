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
        await self.queue.put(job_id)

    async def run(self) -> None:
        while True:
            job_id = await self.queue.get()
            try:
                if job_id is None:
                    return
                await self._process(job_id)
            finally:
                self.queue.task_done()

    async def _process(self, job_id: str) -> None:
        job = await self.store.get(job_id)
        if job is None:
            return
        await self.store.set_status(job_id, JobStatus.RUNNING)
        try:
            result = await self.runtime.predict(job)
            result_dir = self.store.result_dir / job_id
            path = await asyncio.to_thread(export_result, result, job.config, result_dir)
            await self.store.set_status(job_id, JobStatus.SUCCEEDED, result_path=str(path))
        except Exception as exc:
            await self.store.set_status(job_id, JobStatus.FAILED, error=str(exc))
