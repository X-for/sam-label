import pytest
from pydantic import ValidationError

from sam_api.schemas import (
    AggregationPolicy,
    JobCreate,
    JobProgress,
    JobRecord,
    JobStatus,
    PromptGroup,
    TaskType,
    UploadedImage,
)


def test_rejects_duplicate_labels():
    with pytest.raises(ValidationError):
        JobCreate(
            prompt_groups=[
                PromptGroup(label="vehicle", prompts=["car"]),
                PromptGroup(label="vehicle", prompts=["truck"]),
            ]
        )


def test_detect_rejects_union():
    with pytest.raises(ValidationError):
        JobCreate(
            task_type=TaskType.DETECT,
            prompt_groups=[
                PromptGroup(label="vehicle", prompts=["car"], aggregation=AggregationPolicy.UNION)
            ],
        )


def test_seg_alias_and_yolo_class_ids():
    job = JobCreate(
        task_type="seg",
        output_format="yolo",
        prompt_groups=[PromptGroup(label="vehicle", class_id=0, prompts=["car"])],
    )
    assert job.task_type == TaskType.SEGMENT

    with pytest.raises(ValidationError):
        JobCreate(
            output_format="yolo",
            prompt_groups=[PromptGroup(label="vehicle", class_id=2, prompts=["car"])],
        )


def test_prediction_imgsz_must_match_sam3_stride():
    job = JobCreate(prompt_groups=[PromptGroup(label="person", prompts=["person"])])
    assert job.prediction.imgsz == 644

    with pytest.raises(ValidationError, match="multiple of SAM3 stride 14"):
        JobCreate(
            prediction={"imgsz": 640},
            prompt_groups=[PromptGroup(label="person", prompts=["person"])],
        )


def test_succeeded_legacy_job_progress_is_complete():
    record = JobRecord(
        id="legacy-job",
        status=JobStatus.SUCCEEDED,
        config=JobCreate(prompt_groups=[PromptGroup(label="person", prompts=["person"])]),
        images=[
            UploadedImage(
                image_id="image-1",
                original_name="sample.jpg",
                stored_path="/data/sample.jpg",
                size_bytes=1,
                content_type="image/jpeg",
            )
        ],
        processed_images=0,
    )

    progress = JobProgress.from_record(record)

    assert progress.processed_images == 1
    assert progress.total_images == 1
    assert progress.progress_percent == 100.0
