#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def validate_label(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        values = line.split()
        if len(values) != 5 or values[0] != "0":
            raise SystemExit(f"invalid YOLO line at {path}:{line_number}: {line}")
        coordinates = [float(value) for value in values[1:]]
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            raise SystemExit(f"out-of-range YOLO coordinate at {path}:{line_number}: {line}")
        if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
            raise SystemExit(f"empty YOLO box at {path}:{line_number}: {line}")
    return len(lines)


def stable_key(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and index the organized YOLO dataset.")
    parser.add_argument("datasets", type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    args = parser.parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        raise SystemExit("--val-ratio must be between 0 and 1")

    positives: list[str] = []
    negatives: list[str] = []
    annotations = 0
    per_dataset: dict[str, dict[str, int]] = {}
    for dataset_dir in sorted(path for path in args.datasets.iterdir() if path.is_dir()):
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue
        images = sorted(
            (path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda path: path.name.casefold(),
        )
        label_names = {path.stem.casefold(): path for path in labels_dir.glob("*.txt")}
        if len(images) != len(label_names):
            raise SystemExit(
                f"image/label count mismatch in {dataset_dir}: {len(images)} images, {len(label_names)} labels"
            )
        dataset_annotations = 0
        dataset_positives = 0
        for image in images:
            label = label_names.get(image.stem.casefold())
            if label is None:
                raise SystemExit(f"label missing for {image}")
            count = validate_label(label)
            relative = image.relative_to(args.datasets).as_posix()
            annotations += count
            dataset_annotations += count
            if count:
                positives.append(relative)
                dataset_positives += 1
            else:
                negatives.append(relative)
        per_dataset[dataset_dir.name] = {
            "images": len(images),
            "positive_images": dataset_positives,
            "annotations": dataset_annotations,
        }

    train: list[str] = []
    val: list[str] = []
    for group in (positives, negatives):
        ordered = sorted(group, key=stable_key)
        val_count = round(len(ordered) * args.val_ratio)
        val.extend(ordered[:val_count])
        train.extend(ordered[val_count:])
    train.sort(key=str.casefold)
    val.sort(key=str.casefold)
    (args.datasets / "train.txt").write_text(
        "\n".join(f"./{path}" for path in train) + "\n", encoding="utf-8", newline="\n"
    )
    (args.datasets / "val.txt").write_text(
        "\n".join(f"./{path}" for path in val) + "\n", encoding="utf-8", newline="\n"
    )
    (args.datasets / "data.yaml").write_text(
        "train: train.txt\nval: val.txt\nnames:\n  0: explosive_circle_sign\n",
        encoding="utf-8",
        newline="\n",
    )
    summary = {
        "images": len(positives) + len(negatives),
        "positive_images": len(positives),
        "negative_images": len(negatives),
        "annotations": annotations,
        "train_images": len(train),
        "val_images": len(val),
        "class_names": {"0": "explosive_circle_sign"},
        "per_dataset": per_dataset,
    }
    (args.datasets / "preannotation-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
