from __future__ import annotations

import json
import zipfile
from pathlib import Path

from .schemas import Annotation, JobCreate, JobResult, OutputFormat, TaskType


def category_ids(config: JobCreate) -> dict[str, int]:
    return {
        group.label: group.class_id if group.class_id is not None else index
        for index, group in enumerate(config.prompt_groups)
    }


def polygon_area(polygon: list[float]) -> float:
    if len(polygon) < 6:
        return 0.0
    points = list(zip(polygon[0::2], polygon[1::2]))
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )
    ) / 2.0


def export_result(result: JobResult, config: JobCreate, result_dir: Path) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    if config.output_format == OutputFormat.COCO:
        return export_coco(result, config, result_dir)
    return export_yolo(result, config, result_dir)


def export_coco(result: JobResult, config: JobCreate, result_dir: Path) -> Path:
    ids = category_ids(config)
    coco_images = []
    coco_annotations = []
    annotation_id = 1
    for image_number, image in enumerate(result.images, start=1):
        coco_images.append(
            {
                "id": image_number,
                "file_name": image.file_name,
                "width": image.width,
                "height": image.height,
                "sam3_image_id": image.image_id,
            }
        )
        for annotation in image.annotations:
            segmentation = annotation.segmentation or []
            area = (
                sum(polygon_area(polygon) for polygon in segmentation)
                if segmentation
                else annotation.bbox[2] * annotation.bbox[3]
            )
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_number,
                    "category_id": ids[annotation.label],
                    "bbox": annotation.bbox,
                    "area": round(area, 3),
                    "segmentation": segmentation,
                    "iscrowd": 0,
                    "score": annotation.score,
                    "source_prompts": annotation.source_prompts,
                    "instance_count": annotation.instance_count,
                }
            )
            annotation_id += 1
    payload = {
        "info": {
            "description": "SAM3 pre-annotations",
            "task_type": config.task_type.value,
            **result.meta,
        },
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [
            {"id": ids[group.label], "name": group.label, "supercategory": ""}
            for group in config.prompt_groups
        ],
    }
    path = result_dir / "annotations.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    return path


def _normalized_bbox(annotation: Annotation, width: int, height: int) -> list[float] | None:
    x, y, box_width, box_height = annotation.bbox
    x1 = min(float(width), max(0.0, x))
    y1 = min(float(height), max(0.0, y))
    x2 = min(float(width), max(0.0, x + box_width))
    y2 = min(float(height), max(0.0, y + box_height))
    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0.0 or clipped_height <= 0.0:
        return None
    return [
        (x1 + clipped_width / 2.0) / width,
        (y1 + clipped_height / 2.0) / height,
        clipped_width / width,
        clipped_height / height,
    ]


def _format_number(value: float) -> str:
    return f"{min(1.0, max(0.0, value)):.6f}"


def export_yolo(result: JobResult, config: JobCreate, result_dir: Path) -> Path:
    ids = category_ids(config)
    labels_dir = result_dir / "labels"
    labels_dir.mkdir(exist_ok=True)
    used_names: set[str] = set()
    manifest_images = []

    for image in result.images:
        stem = Path(image.file_name).stem
        label_name = f"{stem}.txt"
        if label_name.casefold() in used_names:
            label_name = f"{stem}_{image.image_id[:8]}.txt"
        used_names.add(label_name.casefold())
        lines: list[str] = []
        for annotation in image.annotations:
            class_id = ids[annotation.label]
            if config.task_type == TaskType.DETECT:
                coordinates = _normalized_bbox(annotation, image.width, image.height)
                if coordinates is None:
                    continue
                lines.append(f"{class_id} " + " ".join(_format_number(value) for value in coordinates))
            else:
                for polygon in annotation.segmentation or []:
                    normalized = [
                        value / (image.width if index % 2 == 0 else image.height)
                        for index, value in enumerate(polygon)
                    ]
                    if len(normalized) >= 6:
                        lines.append(f"{class_id} " + " ".join(_format_number(value) for value in normalized))
        (labels_dir / label_name).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n"
        )
        manifest_images.append(
            {
                "image_id": image.image_id,
                "image_file": image.file_name,
                "label_file": f"labels/{label_name}",
                "width": image.width,
                "height": image.height,
                "predictions": [annotation.model_dump(mode="json") for annotation in image.annotations],
            }
        )

    names = {ids[group.label]: group.label for group in config.prompt_groups}
    yaml_lines = ["path: .", "train: images", "val: images", f"task: {config.task_type.value}", "names:"]
    yaml_lines.extend(f"  {class_id}: {json.dumps(name, ensure_ascii=False)}" for class_id, name in sorted(names.items()))
    (result_dir / "data.yaml").write_text(
        "\n".join(yaml_lines) + "\n", encoding="utf-8", newline="\n"
    )
    (result_dir / "manifest.json").write_text(
        json.dumps(
            {"job_id": result.job_id, "task_type": config.task_type.value, "images": manifest_images},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    archive = result_dir / "yolo-labels.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.write(result_dir / "data.yaml", "data.yaml")
        output.write(result_dir / "manifest.json", "manifest.json")
        for label_path in sorted(labels_dir.glob("*.txt")):
            output.write(label_path, f"labels/{label_path.name}")
    return archive
