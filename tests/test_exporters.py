import json
import zipfile

from sam_api.exporters import export_result
from sam_api.schemas import (
    Annotation,
    ImageResult,
    JobCreate,
    JobResult,
    OutputFormat,
    PromptGroup,
    TaskType,
)


def sample_result(task_type):
    return JobResult(
        job_id="job-1",
        task_type=task_type,
        images=[
            ImageResult(
                image_id="image-1",
                file_name="sample.jpg",
                width=100,
                height=50,
                annotations=[
                    Annotation(
                        id=1,
                        label="car",
                        class_id=None,
                        score=0.9,
                        bbox=[10, 5, 20, 10],
                        segmentation=[[10, 5, 30, 5, 30, 15, 10, 15]],
                        source_prompts=["car"],
                    )
                ],
            )
        ],
    )


def test_coco_export(tmp_path):
    config = JobCreate(task_type="seg", output_format="coco", prompt_groups=[PromptGroup(label="car", prompts=["car"])])
    path = export_result(sample_result(TaskType.SEGMENT), config, tmp_path)
    payload = json.loads(path.read_text("utf-8"))
    assert path.name == "annotations.json"
    assert payload["annotations"][0]["bbox"] == [10.0, 5.0, 20.0, 10.0]
    assert payload["categories"] == [{"id": 0, "name": "car", "supercategory": ""}]


def test_yolo_detect_export(tmp_path):
    config = JobCreate(task_type="detect", output_format="yolo", prompt_groups=[PromptGroup(label="car", prompts=["car"])])
    path = export_result(sample_result(TaskType.DETECT), config, tmp_path)
    with zipfile.ZipFile(path) as archive:
        label = archive.read("labels/sample.txt").decode()
        assert label == "0 0.200000 0.200000 0.200000 0.200000\n"
        assert "manifest.json" in archive.namelist()


def test_yolo_detect_export_clips_box_to_image_boundaries(tmp_path):
    result = sample_result(TaskType.DETECT)
    result.images[0].annotations[0].bbox = [95, -5, 10, 20]
    config = JobCreate(
        task_type="detect",
        output_format="yolo",
        prompt_groups=[PromptGroup(label="car", prompts=["car"])],
    )

    path = export_result(result, config, tmp_path)

    with zipfile.ZipFile(path) as archive:
        label = archive.read("labels/sample.txt").decode()
    assert label == "0 0.975000 0.150000 0.050000 0.300000\n"


def test_yolo_detect_export_skips_box_outside_image(tmp_path):
    result = sample_result(TaskType.DETECT)
    result.images[0].annotations[0].bbox = [105, 5, 10, 20]
    config = JobCreate(
        task_type="detect",
        output_format="yolo",
        prompt_groups=[PromptGroup(label="car", prompts=["car"])],
    )

    path = export_result(result, config, tmp_path)

    with zipfile.ZipFile(path) as archive:
        assert archive.read("labels/sample.txt") == b""
