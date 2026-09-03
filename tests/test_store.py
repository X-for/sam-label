import asyncio
import json

from sam_api.schemas import JobCreate, JobRecord, JobStatus, PromptGroup, UploadedImage
from sam_api.store import JobStore


def test_initialize_backfills_progress_for_legacy_succeeded_job(tmp_path):
    async def exercise():
        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()
        record = JobRecord(
            id="legacy-job",
            status=JobStatus.SUCCEEDED,
            config=JobCreate(
                prompt_groups=[PromptGroup(label="person", prompts=["person"])]
            ),
            images=[
                UploadedImage(
                    image_id="image-1",
                    original_name="sample.jpg",
                    stored_path=str(tmp_path / "uploads" / "sample.jpg"),
                    size_bytes=1,
                    content_type="image/jpeg",
                )
            ],
        )
        payload = record.model_dump(mode="json")
        payload.pop("processed_images")
        metadata_path = jobs_dir / "legacy-job.json"
        metadata_path.write_text(json.dumps(payload), encoding="utf-8")

        store = JobStore(tmp_path)
        await store.initialize()

        loaded = await store.get("legacy-job")
        assert loaded is not None
        assert loaded.processed_images == 1
        persisted = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert persisted["processed_images"] == 1

    asyncio.run(exercise())
