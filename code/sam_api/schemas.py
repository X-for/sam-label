from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TaskType(StrEnum):
    DETECT = "detect"
    SEGMENT = "segment"
    SEMANTIC = "semantic"


class OutputFormat(StrEnum):
    COCO = "coco"
    YOLO = "yolo"


class AggregationPolicy(StrEnum):
    DEDUPLICATE = "deduplicate"
    KEEP_ALL = "keep_all"
    BEST = "best"
    UNION = "union"


class JobStatus(StrEnum):
    UPLOADING = "uploading"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class PromptGroup(BaseModel):
    """Several text prompts that all map to one output label."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=128)
    prompts: list[str] = Field(min_length=1, max_length=64)
    class_id: int | None = Field(default=None, ge=0)
    aggregation: AggregationPolicy = AggregationPolicy.DEDUPLICATE
    merge_iou: float = Field(default=0.7, ge=0.0, le=1.0)

    @field_validator("label")
    @classmethod
    def clean_label(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("label cannot be blank")
        return value

    @field_validator("prompts")
    @classmethod
    def clean_prompts(cls, values: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(value.strip() for value in values if value.strip()))
        if not cleaned:
            raise ValueError("at least one non-blank prompt is required")
        return cleaned


class PredictionParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conf: float = Field(default=0.25, ge=0.0, le=1.0)
    iou: float = Field(default=0.7, ge=0.0, le=1.0)
    imgsz: int = Field(default=644, ge=64, le=4096)
    max_det: int = Field(default=300, ge=1, le=5000)
    retina_masks: bool = True

    @field_validator("imgsz")
    @classmethod
    def validate_imgsz(cls, value: int) -> int:
        if value % 14:
            lower = value - value % 14
            upper = lower + 14
            raise ValueError(f"imgsz must be a multiple of SAM3 stride 14; try {lower} or {upper}")
        return value


class JobCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: TaskType = TaskType.SEGMENT
    output_format: OutputFormat = OutputFormat.COCO
    prompt_groups: list[PromptGroup] = Field(min_length=1, max_length=256)
    prediction: PredictionParams = Field(default_factory=PredictionParams)
    client_reference: str | None = Field(default=None, max_length=256)

    @field_validator("task_type", mode="before")
    @classmethod
    def task_aliases(cls, value: Any) -> Any:
        aliases = {"seg": "segment", "instance_seg": "segment", "det": "detect"}
        return aliases.get(value, value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_config(self) -> "JobCreate":
        labels = [group.label for group in self.prompt_groups]
        if len(labels) != len(set(labels)):
            raise ValueError("each label must appear in exactly one prompt group")
        class_ids = [group.class_id for group in self.prompt_groups if group.class_id is not None]
        if len(class_ids) != len(set(class_ids)):
            raise ValueError("class_id values must be unique")
        if class_ids and len(class_ids) != len(self.prompt_groups):
            raise ValueError("class_id must be provided for every prompt group or for none of them")
        if self.output_format == OutputFormat.YOLO and class_ids:
            expected = list(range(len(self.prompt_groups)))
            if sorted(class_ids) != expected:
                raise ValueError(f"YOLO class_id values must be contiguous and 0-based: {expected}")
        if self.task_type == TaskType.DETECT and any(
            group.aggregation == AggregationPolicy.UNION for group in self.prompt_groups
        ):
            raise ValueError("union aggregation is not valid for detect tasks")
        return self


class UploadedImage(BaseModel):
    image_id: str
    original_name: str
    stored_path: str
    size_bytes: int
    content_type: str | None = None


class JobRecord(BaseModel):
    id: str
    status: JobStatus
    config: JobCreate
    images: list[UploadedImage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    processed_images: int = Field(default=0, ge=0)
    error: str | None = None
    result_path: str | None = None


class JobView(BaseModel):
    id: str
    status: JobStatus
    client_reference: str | None
    output_format: OutputFormat
    image_count: int
    uploaded_bytes: int
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    result_url: str | None = None

    @classmethod
    def from_record(cls, record: JobRecord) -> "JobView":
        return cls(
            id=record.id,
            status=record.status,
            client_reference=record.config.client_reference,
            output_format=record.config.output_format,
            image_count=len(record.images),
            uploaded_bytes=sum(image.size_bytes for image in record.images),
            created_at=record.created_at,
            updated_at=record.updated_at,
            error=record.error,
            result_url=f"/v1/jobs/{record.id}/result" if record.result_path else None,
        )


class JobList(BaseModel):
    items: list[JobView]
    total: int
    limit: int
    offset: int


class JobProgress(BaseModel):
    id: str
    status: JobStatus
    processed_images: int
    total_images: int
    progress_percent: float
    updated_at: datetime
    error: str | None = None

    @classmethod
    def from_record(cls, record: JobRecord) -> "JobProgress":
        total_images = len(record.images)
        if record.status == JobStatus.SUCCEEDED:
            processed_images = total_images
            progress_percent = 100.0
        else:
            processed_images = min(record.processed_images, total_images)
            progress_percent = (
                round(processed_images * 100.0 / total_images, 2) if total_images else 0.0
            )
        return cls(
            id=record.id,
            status=record.status,
            processed_images=processed_images,
            total_images=total_images,
            progress_percent=progress_percent,
            updated_at=record.updated_at,
            error=record.error,
        )


class Annotation(BaseModel):
    id: int
    label: str
    class_id: int | None
    score: float
    bbox: list[float] = Field(description="COCO xywh coordinates")
    segmentation: list[list[float]] | None = None
    source_prompts: list[str]
    instance_count: int = 1


class ImageResult(BaseModel):
    image_id: str
    file_name: str
    width: int
    height: int
    annotations: list[Annotation]


class JobResult(BaseModel):
    job_id: str
    task_type: TaskType
    images: list[ImageResult]
    meta: dict[str, Any] = Field(default_factory=dict)


class ModelStatus(BaseModel):
    loaded: bool
    loading: bool
    lifecycle: str
    device: str
    model_path: str
    last_error: str | None = None
