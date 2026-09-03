from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from uuid import uuid4

from .schemas import (
    ExpectedUpload,
    JobCreate,
    JobRecord,
    JobStatus,
    UploadManifestFile,
    UploadManifestStatus,
    UploadedImage,
    utc_now,
)


class JobStore:
    """In-memory state plus durable per-job metadata and result artifacts."""

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._metadata_dir = data_dir / "jobs"
        self._manifest_dir = data_dir / "manifests"
        self._jobs: dict[str, JobRecord] = {}
        self._expected_by_path: dict[str, dict[str, ExpectedUpload]] = {}
        self._uploaded_by_path: dict[str, dict[str, UploadedImage]] = {}
        self._uploaded_ids: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()

    @property
    def result_dir(self) -> Path:
        return self._data_dir / "results"

    async def initialize(self) -> None:
        self._metadata_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_dir.mkdir(parents=True, exist_ok=True)
        for path in self._metadata_dir.glob("*.json"):
            try:
                record = JobRecord.model_validate_json(await asyncio.to_thread(path.read_text, "utf-8"))
            except Exception:
                continue
            receipt_path = self._receipt_path(record.id)
            if receipt_path.exists():
                record.images = await self._read_images(receipt_path)
                await self._write_images(receipt_path, record.images)
            elif record.images:
                await self._append_images(receipt_path, record.images)
            manifest_path = self._manifest_path(record.id)
            if manifest_path.exists():
                record.expected_images = await self._read_manifest(manifest_path)
                record.expected_image_count = len(record.expected_images)
                if record.status == JobStatus.UPLOADING:
                    await self._reconcile_upload(record)
            record.image_count = len(record.images)
            record.uploaded_bytes = sum(image.size_bytes for image in record.images)
            if record.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                record.status = JobStatus.FAILED
                record.error = "service restarted before this job completed"
            elif record.status == JobStatus.SUCCEEDED:
                # Progress tracking was added after some completed jobs had
                # already been persisted. Keep the durable record consistent
                # with the terminal status when those jobs are loaded.
                record.processed_images = len(record.images)
            self._jobs[record.id] = record
            self._reindex(record)
            await self._persist(record)

    async def create(self, config: JobCreate) -> JobRecord:
        record = JobRecord(id=str(uuid4()), status=JobStatus.UPLOADING, config=config)
        async with self._lock:
            self._jobs[record.id] = record
            self._reindex(record)
            await self._persist(record)
        return record.model_copy(deep=True)

    async def get(self, job_id: str, *, deep: bool = True) -> JobRecord | None:
        async with self._lock:
            record = self._jobs.get(job_id)
            return record.model_copy(deep=deep) if record else None

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
            return [record.model_copy(deep=False) for record in page], total

    async def add_image(self, job_id: str, image: UploadedImage) -> JobRecord:
        return await self.add_images(job_id, [image])

    async def add_images(self, job_id: str, images: list[UploadedImage]) -> JobRecord:
        async with self._lock:
            record = self._jobs[job_id]
            existing_ids = self._uploaded_ids[job_id]
            existing_paths = self._uploaded_by_path[job_id]
            batch_ids: set[str] = set()
            batch_paths: set[str] = set()
            new_images: list[UploadedImage] = []
            for image in images:
                if image.image_id in existing_ids or image.image_id in batch_ids or (
                    image.relative_path is not None
                    and (
                        image.relative_path in existing_paths
                        or image.relative_path in batch_paths
                    )
                ):
                    continue
                new_images.append(image)
                batch_ids.add(image.image_id)
                if image.relative_path is not None:
                    batch_paths.add(image.relative_path)
            if not new_images:
                return record.model_copy(deep=False)
            await self._append_images(self._receipt_path(job_id), new_images)
            record.images.extend(new_images)
            existing_ids.update(image.image_id for image in new_images)
            existing_paths.update(
                (image.relative_path, image)
                for image in new_images
                if image.relative_path is not None
            )
            record.image_count += len(new_images)
            record.uploaded_bytes += sum(image.size_bytes for image in new_images)
            record.updated_at = utc_now()
            await self._persist(record)
            return record.model_copy(deep=False)

    async def set_manifest(
        self, job_id: str, files: list[UploadManifestFile]
    ) -> UploadManifestStatus:
        async with self._lock:
            record = self._jobs[job_id]
            if record.status != JobStatus.UPLOADING:
                raise ValueError("job no longer accepts an upload manifest")
            if record.images:
                raise ValueError("upload manifest must be registered before uploading images")
            expected = [
                ExpectedUpload(image_id=str(uuid4()), **file.model_dump())
                for file in files
            ]
            await self._write_manifest(self._manifest_path(job_id), expected)
            record.expected_images = expected
            record.expected_image_count = len(expected)
            self._expected_by_path[job_id] = {
                item.relative_path: item for item in expected
            }
            record.updated_at = utc_now()
            await self._persist(record)
            return self._manifest_status(record)

    async def get_manifest_status(self, job_id: str) -> UploadManifestStatus:
        async with self._lock:
            return self._manifest_status(self._jobs[job_id])

    async def missing_files(self, job_id: str) -> list[str]:
        async with self._lock:
            return self._manifest_status(self._jobs[job_id]).missing_files

    async def match_uploads(
        self, job_id: str, paths: list[str]
    ) -> list[tuple[ExpectedUpload | None, UploadedImage | None]]:
        async with self._lock:
            expected = self._expected_by_path[job_id]
            uploaded = self._uploaded_by_path[job_id]
            return [(expected.get(path), uploaded.get(path)) for path in paths]

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
            return record.model_copy(deep=False)

    async def set_progress(self, job_id: str, processed_images: int) -> JobRecord:
        async with self._lock:
            record = self._jobs[job_id]
            record.processed_images = min(max(0, processed_images), len(record.images))
            record.updated_at = utc_now()
            await self._persist(record)
            return record.model_copy(deep=False)

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

    def _receipt_path(self, job_id: str) -> Path:
        return self._metadata_dir / f"{job_id}.images.jsonl"

    def _manifest_path(self, job_id: str) -> Path:
        return self._manifest_dir / f"{job_id}.json"

    async def _append_images(self, path: Path, images: list[UploadedImage]) -> None:
        payload = "".join(
            f"{image.model_dump_json()}\n"
            for image in images
        )

        def append() -> None:
            with path.open("a", encoding="utf-8") as output:
                output.write(payload)
                output.flush()

        await asyncio.to_thread(append)

    async def _read_images(self, path: Path) -> list[UploadedImage]:
        contents = await asyncio.to_thread(path.read_text, "utf-8")
        images: list[UploadedImage] = []
        for line in contents.splitlines():
            if not line.strip():
                continue
            try:
                images.append(UploadedImage.model_validate_json(line))
            except Exception:
                continue
        return images

    async def _write_images(self, path: Path, images: list[UploadedImage]) -> None:
        temporary = path.with_suffix(".tmp")
        payload = "".join(f"{image.model_dump_json()}\n" for image in images)

        def write() -> None:
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(path)

        await asyncio.to_thread(write)

    async def _reconcile_upload(self, record: JobRecord) -> None:
        upload_dir = self._data_dir / "uploads" / record.id
        if upload_dir.exists():
            for partial in upload_dir.glob("*.part"):
                await asyncio.to_thread(partial.unlink, True)

        receipts_by_id = {image.image_id: image for image in record.images}
        reconciled: list[UploadedImage] = []
        for expected in record.expected_images:
            suffix = Path(expected.relative_path).suffix.lower()[:10]
            destination = upload_dir / f"{expected.image_id}{suffix}"
            receipt = receipts_by_id.get(expected.image_id)
            candidate = Path(receipt.stored_path) if receipt is not None else destination
            try:
                valid = candidate.is_file() and candidate.stat().st_size == expected.size_bytes
            except OSError:
                valid = False
            if not valid:
                continue
            reconciled.append(
                UploadedImage(
                    image_id=expected.image_id,
                    original_name=Path(expected.relative_path).name,
                    relative_path=expected.relative_path,
                    stored_path=str(candidate.resolve()),
                    size_bytes=expected.size_bytes,
                    content_type=(receipt.content_type if receipt else expected.content_type),
                )
            )
        if reconciled != record.images:
            record.images = reconciled
            await self._write_images(self._receipt_path(record.id), reconciled)

    async def _write_manifest(self, path: Path, files: list[ExpectedUpload]) -> None:
        temporary = path.with_suffix(".tmp")
        payload = json.dumps(
            [file.model_dump(mode="json") for file in files],
            ensure_ascii=False,
            separators=(",", ":"),
        )

        def write() -> None:
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(path)

        await asyncio.to_thread(write)

    async def _read_manifest(self, path: Path) -> list[ExpectedUpload]:
        payload = json.loads(await asyncio.to_thread(path.read_text, "utf-8"))
        return [ExpectedUpload.model_validate(item) for item in payload]

    @staticmethod
    def _manifest_status(record: JobRecord) -> UploadManifestStatus:
        uploaded_by_path = {
            image.relative_path: image
            for image in record.images
            if image.relative_path is not None
        }
        missing: list[str] = []
        for expected in record.expected_images:
            image = uploaded_by_path.get(expected.relative_path)
            if image is None:
                missing.append(expected.relative_path)
                continue
            try:
                valid = (
                    image.image_id == expected.image_id
                    and Path(image.stored_path).is_file()
                    and Path(image.stored_path).stat().st_size == expected.size_bytes
                )
            except OSError:
                valid = False
            if not valid:
                missing.append(expected.relative_path)
        return UploadManifestStatus(
            expected_image_count=record.expected_image_count,
            uploaded_image_count=(
                record.expected_image_count - len(missing)
                if record.expected_images
                else record.image_count
            ),
            missing_files=missing,
        )

    def _reindex(self, record: JobRecord) -> None:
        self._expected_by_path[record.id] = {
            item.relative_path: item for item in record.expected_images
        }
        self._uploaded_by_path[record.id] = {
            image.relative_path: image
            for image in record.images
            if image.relative_path is not None
        }
        self._uploaded_ids[record.id] = {image.image_id for image in record.images}

    async def save_result(self, job_id: str, payload: dict) -> Path:
        result_dir = self._data_dir / "results" / job_id
        result_dir.mkdir(parents=True, exist_ok=True)
        path = result_dir / "result.json"
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        await asyncio.to_thread(path.write_text, serialized, "utf-8")
        return path
