import asyncio
import json

from sam_api.schemas import (
    JobCreate,
    JobRecord,
    JobStatus,
    PromptGroup,
    UploadManifestFile,
    UploadedImage,
)
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
        payload["images"] = [image.model_dump(mode="json") for image in record.images]
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


def test_add_images_persists_a_slim_job_and_restores_the_batch(tmp_path):
    async def exercise():
        store = JobStore(tmp_path)
        await store.initialize()
        job = await store.create(
            JobCreate(prompt_groups=[PromptGroup(label="person", prompts=["person"])])
        )
        images = [
            UploadedImage(
                image_id="image-1",
                original_name="first.jpg",
                stored_path=str(tmp_path / "uploads" / job.id / "image-1.jpg"),
                size_bytes=11,
                content_type="image/jpeg",
            ),
            UploadedImage(
                image_id="image-2",
                original_name="second.jpg",
                stored_path=str(tmp_path / "uploads" / job.id / "image-2.jpg"),
                size_bytes=13,
                content_type="image/jpeg",
            ),
        ]

        saved = await store.add_images(job.id, images)

        assert [image.image_id for image in saved.images] == ["image-1", "image-2"]
        metadata = json.loads((tmp_path / "jobs" / f"{job.id}.json").read_text("utf-8"))
        assert "images" not in metadata
        assert metadata["image_count"] == 2
        assert metadata["uploaded_bytes"] == 24
        receipt_lines = (tmp_path / "jobs" / f"{job.id}.images.jsonl").read_text(
            "utf-8"
        ).splitlines()
        assert len(receipt_lines) == 2

        restored_store = JobStore(tmp_path)
        await restored_store.initialize()
        restored = await restored_store.get(job.id)
        assert restored is not None
        assert [image.original_name for image in restored.images] == [
            "first.jpg",
            "second.jpg",
        ]

    asyncio.run(exercise())


def test_initialize_migrates_legacy_embedded_images_to_receipt_log(tmp_path):
    async def exercise():
        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()
        record = JobRecord(
            id="legacy-upload",
            status=JobStatus.UPLOADING,
            config=JobCreate(
                prompt_groups=[PromptGroup(label="person", prompts=["person"])]
            ),
            images=[
                UploadedImage(
                    image_id="image-1",
                    original_name="sample.jpg",
                    stored_path=str(tmp_path / "uploads" / "legacy-upload" / "image-1.jpg"),
                    size_bytes=7,
                    content_type="image/jpeg",
                )
            ],
        )
        metadata_path = jobs_dir / "legacy-upload.json"
        payload = record.model_dump(mode="json")
        payload["images"] = [image.model_dump(mode="json") for image in record.images]
        metadata_path.write_text(json.dumps(payload), encoding="utf-8")

        store = JobStore(tmp_path)
        await store.initialize()

        metadata = json.loads(metadata_path.read_text("utf-8"))
        assert "images" not in metadata
        assert metadata["image_count"] == 1
        receipt = jobs_dir / "legacy-upload.images.jsonl"
        assert json.loads(receipt.read_text("utf-8"))["original_name"] == "sample.jpg"
        restored = await store.get("legacy-upload")
        assert restored is not None
        assert len(restored.images) == 1

    asyncio.run(exercise())


def test_initialize_recovers_completed_manifest_file_without_receipt(tmp_path):
    async def exercise():
        store = JobStore(tmp_path)
        await store.initialize()
        job = await store.create(
            JobCreate(prompt_groups=[PromptGroup(label="person", prompts=["person"])])
        )
        await store.set_manifest(
            job.id,
            [UploadManifestFile(relative_path="dataset/sample.jpg", size_bytes=4)],
        )
        prepared = await store.get(job.id)
        assert prepared is not None
        image_id = prepared.expected_images[0].image_id
        upload_dir = tmp_path / "uploads" / job.id
        upload_dir.mkdir(parents=True)
        (upload_dir / f"{image_id}.jpg").write_bytes(b"data")

        restored_store = JobStore(tmp_path)
        await restored_store.initialize()

        restored = await restored_store.get(job.id)
        assert restored is not None
        assert restored.image_count == 1
        assert restored.images[0].relative_path == "dataset/sample.jpg"
        assert (await restored_store.get_manifest_status(job.id)).missing_files == []

    asyncio.run(exercise())


def test_initialize_marks_missing_receipt_file_for_retry(tmp_path):
    async def exercise():
        store = JobStore(tmp_path)
        await store.initialize()
        job = await store.create(
            JobCreate(prompt_groups=[PromptGroup(label="person", prompts=["person"])])
        )
        await store.set_manifest(
            job.id,
            [UploadManifestFile(relative_path="sample.jpg", size_bytes=4)],
        )
        prepared = await store.get(job.id)
        assert prepared is not None
        expected = prepared.expected_images[0]
        await store.add_images(
            job.id,
            [
                UploadedImage(
                    image_id=expected.image_id,
                    original_name="sample.jpg",
                    relative_path="sample.jpg",
                    stored_path=str(tmp_path / "uploads" / job.id / f"{expected.image_id}.jpg"),
                    size_bytes=4,
                    content_type="image/jpeg",
                )
            ],
        )

        restored_store = JobStore(tmp_path)
        await restored_store.initialize()

        restored = await restored_store.get(job.id)
        assert restored is not None
        assert restored.image_count == 0
        assert (await restored_store.get_manifest_status(job.id)).missing_files == [
            "sample.jpg"
        ]

    asyncio.run(exercise())


def test_receipt_log_recovers_after_an_incomplete_final_line(tmp_path):
    async def exercise():
        store = JobStore(tmp_path)
        await store.initialize()
        job = await store.create(
            JobCreate(prompt_groups=[PromptGroup(label="person", prompts=["person"])])
        )
        first = UploadedImage(
            image_id="image-1",
            original_name="first.jpg",
            stored_path=str(tmp_path / "uploads" / job.id / "image-1.jpg"),
            size_bytes=1,
            content_type="image/jpeg",
        )
        await store.add_images(job.id, [first])
        receipt = tmp_path / "jobs" / f"{job.id}.images.jsonl"
        with receipt.open("a", encoding="utf-8") as output:
            output.write('{"image_id":"interrupted"')

        restored_store = JobStore(tmp_path)
        await restored_store.initialize()
        second = UploadedImage(
            image_id="image-2",
            original_name="second.jpg",
            stored_path=str(tmp_path / "uploads" / job.id / "image-2.jpg"),
            size_bytes=1,
            content_type="image/jpeg",
        )
        await restored_store.add_images(job.id, [second])

        final_store = JobStore(tmp_path)
        await final_store.initialize()
        restored = await final_store.get(job.id)
        assert restored is not None
        assert [image.image_id for image in restored.images] == ["image-1", "image-2"]

    asyncio.run(exercise())


def test_add_images_deduplicates_repeated_entries_inside_one_batch(tmp_path):
    async def exercise():
        store = JobStore(tmp_path)
        await store.initialize()
        job = await store.create(
            JobCreate(prompt_groups=[PromptGroup(label="person", prompts=["person"])])
        )
        image = UploadedImage(
            image_id="same-id",
            original_name="same.jpg",
            relative_path="dataset/same.jpg",
            stored_path=str(tmp_path / "uploads" / job.id / "same-id.jpg"),
            size_bytes=4,
            content_type="image/jpeg",
        )

        saved = await store.add_images(job.id, [image, image.model_copy(deep=True)])

        assert saved.image_count == 1
        assert len(saved.images) == 1
        receipt = tmp_path / "jobs" / f"{job.id}.images.jsonl"
        assert len(receipt.read_text("utf-8").splitlines()) == 1

    asyncio.run(exercise())
