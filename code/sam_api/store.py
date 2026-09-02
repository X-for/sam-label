from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from uuid import uuid4

from .schemas import JobCreate, JobRecord, JobStatus, UploadedImage, utc_now


class JobStore:
    """In-memory state plus durable per-job metadata and result artifacts."""

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._metadata_dir = data_dir / "jobs"
        self._jobs: dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()

    @property
    def result_dir(self) -> Path:
        return self._data_dir / "results"

    async def initialize(self) -> None:
        self._metadata_dir.mkdir(parents=True, exist_ok=True)
        for path in self._metadata_dir.glob("*.json"):
            try:
                record = JobRecord.model_validate_json(await asyncio.to_thread(path.read_text, "utf-8"))
            except Exception:
                continue
            if record.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                record.status = JobStatus.FAILED
                record.error = "service restarted before this job completed"
            self._jobs[record.id] = record
            await self._persist(record)

    async def create(self, config: JobCreate) -> JobRecord:
        record = JobRecord(id=str(uuid4()), status=JobStatus.UPLOADING, config=config)
        async with self._lock:
            self._jobs[record.id] = record
            await self._persist(record)
        return record.model_copy(deep=True)

    async def get(self, job_id: str) -> JobRecord | None:
        async with self._lock:
            record = self._jobs.get(job_id)
            return record.model_copy(deep=True) if record else None

    async def list_jobs(
        self,
        *,
        status: JobStatus | None,
        limit: int,
        offset: int,
    ) -> tuple[list[JobRecord], int]:
        async with self._lock:
            records = [
                record for record in self._jobs.values() if status is None or record.status == status
            ]
            records.sort(key=lambda record: (record.created_at, record.id), reverse=True)
            total = len(records)
            page = records[offset : offset + limit]
            return [record.model_copy(deep=True) for record in page], total

    async def add_image(self, job_id: str, image: UploadedImage) -> JobRecord:
        async with self._lock:
            record = self._jobs[job_id]
            record.images.append(image)
            record.updated_at = utc_now()
            await self._persist(record)
            return record.model_copy(deep=True)

    async def set_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error: str | None = None,
        result_path: str | None = None,
    ) -> JobRecord:
        async with self._lock:
            record = self._jobs[job_id]
            record.status = status
            record.error = error
            if status == JobStatus.SUCCEEDED:
                record.processed_images = len(record.images)
            if result_path is not None:
                record.result_path = result_path
            record.updated_at = utc_now()
            await self._persist(record)
            return record.model_copy(deep=True)

    async def set_progress(self, job_id: str, processed_images: int) -> JobRecord:
        async with self._lock:
            record = self._jobs[job_id]
            record.processed_images = min(max(0, processed_images), len(record.images))
            record.updated_at = utc_now()
            await self._persist(record)
            return record.model_copy(deep=True)

    async def delete_result(self, job_id: str) -> JobRecord:
        async with self._lock:
            record = self._jobs[job_id]
            result_root = self.result_dir.resolve()
            job_result_dir = (self.result_dir / job_id).resolve()
            if job_result_dir.parent != result_root:
                raise ValueError("job result directory escapes the configured result root")
            await asyncio.to_thread(shutil.rmtree, job_result_dir, True)
            record.result_path = None
            record.updated_at = utc_now()
            await self._persist(record)
            return record.model_copy(deep=True)

    async def _persist(self, record: JobRecord) -> None:
        path = self._metadata_dir / f"{record.id}.json"
        temporary = path.with_suffix(".tmp")
        payload = record.model_dump_json(indent=2)

        def write() -> None:
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(path)

        await asyncio.to_thread(write)

    async def save_result(self, job_id: str, payload: dict) -> Path:
        result_dir = self._data_dir / "results" / job_id
        result_dir.mkdir(parents=True, exist_ok=True)
        path = result_dir / "result.json"
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        await asyncio.to_thread(path.write_text, serialized, "utf-8")
        return path
