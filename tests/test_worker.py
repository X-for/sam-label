import asyncio

from sam_api.config import Settings
from sam_api.model_runtime import Sam3Runtime
from sam_api.worker import JobWorker


class RecordingRuntime:
    def __init__(self):
        self.events = []

    def queue_became_active(self):
        self.events.append("active")

    def queue_became_idle(self):
        self.events.append("idle")


class RecordingWorker(JobWorker):
    def __init__(self, runtime):
        super().__init__(store=None, runtime=runtime)
        self.processed = []

    async def _process(self, job_id):
        self.processed.append(job_id)


def test_queue_activity_wraps_processing():
    async def exercise():
        runtime = RecordingRuntime()
        worker = RecordingWorker(runtime)
        await worker.enqueue("job-1")
        await worker.enqueue("job-2")
        runner = asyncio.create_task(worker.run())
        await worker.queue.join()
        await worker.queue.put(None)
        await runner

        assert worker.processed == ["job-1", "job-2"]
        assert runtime.events == ["active", "active", "idle"]

    asyncio.run(exercise())


def test_loaded_model_is_retained_until_queue_becomes_idle(monkeypatch):
    async def exercise():
        runtime = Sam3Runtime(Settings(idle_unload_seconds=0))
        runtime._predictor = object()
        unloads = []

        async def fake_unload():
            unloads.append(True)
            runtime._predictor = None

        monkeypatch.setattr(runtime, "_unload_locked", fake_unload)
        runtime.queue_became_active()
        await runtime.maybe_unload_idle()
        assert unloads == []

        runtime.queue_became_idle()
        await runtime.maybe_unload_idle()
        assert unloads == [True]
        runtime._executor.shutdown(wait=True, cancel_futures=True)

    asyncio.run(exercise())
