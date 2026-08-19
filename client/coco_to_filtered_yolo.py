#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Filter SAM3 COCO segment candidates and emit YOLO detection labels."
    )
    result.add_argument("coco", type=Path)
    result.add_argument("images_dir", type=Path)
    result.add_argument("--labels-dir", type=Path)
    result.add_argument("--audit", type=Path)
    result.add_argument("--min-score", type=float, default=0.50)
    result.add_argument("--min-fill-ratio", type=float, default=0.70)
    result.add_argument("--min-prompt-votes", type=int, default=2)
    result.add_argument("--min-relative-area", type=float, default=0.002)
    return result


def normalized_box(bbox: list[float], width: int, height: int) -> list[float] | None:
    x, y, box_width, box_height = (float(value) for value in bbox)
    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + box_width))
    y2 = max(0.0, min(float(height), y + box_height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [
        ((x1 + x2) / 2.0) / width,
        ((y1 + y2) / 2.0) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    ]


def reject_reasons(annotation: dict[str, Any], image: dict[str, Any], args: argparse.Namespace) -> tuple[list[str], dict[str, float | int]]:
    bbox = annotation["bbox"]
    box_area = max(0.0, float(bbox[2])) * max(0.0, float(bbox[3]))
    image_area = max(1, int(image["width"]) * int(image["height"]))
    fill_ratio = float(annotation.get("area", 0.0)) / box_area if box_area else 0.0
    relative_area = box_area / image_area
    prompt_votes = len({str(item) for item in annotation.get("source_prompts", [])})
    score = float(annotation.get("score", 0.0))
    reasons: list[str] = []
    if not annotation.get("segmentation"):
        reasons.append("missing_segmentation")
    if score < args.min_score:
        reasons.append("low_score")
    if fill_ratio < args.min_fill_ratio:
        reasons.append("low_fill_ratio")
    if prompt_votes < args.min_prompt_votes:
        reasons.append("low_prompt_votes")
    if relative_area < args.min_relative_area:
        reasons.append("too_small")
    return reasons, {
        "score": round(score, 6),
        "fill_ratio": round(fill_ratio, 6),
        "prompt_votes": prompt_votes,
        "relative_area": round(relative_area, 8),
    }


def main() -> int:
    args = parser().parse_args()
    payload = json.loads(args.coco.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in payload["images"]}
    annotations: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in images}
    for annotation in payload["annotations"]:
        annotations.setdefault(int(annotation["image_id"]), []).append(annotation)

    source_files = {
        item.name.casefold(): item
        for item in args.images_dir.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    }
    if len(source_files) != len([
        item for item in args.images_dir.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    ]):
        raise SystemExit(f"duplicate case-insensitive image names in {args.images_dir}")

    labels_dir = args.labels_dir or args.images_dir.parent / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.audit or args.images_dir.parent / "explosive-circle-sign-audit.json"
    accepted_count = 0
    rejected_count = 0
    audit_images: list[dict[str, Any]] = []

    for image_id, image in images.items():
        file_name = Path(str(image["file_name"])).name
        source = source_files.get(file_name.casefold())
        if source is None:
            raise SystemExit(f"COCO image is missing locally: {file_name}")
        lines: list[str] = []
        decisions: list[dict[str, Any]] = []
        for annotation in annotations.get(image_id, []):
            reasons, metrics = reject_reasons(annotation, image, args)
            coordinates = normalized_box(annotation["bbox"], int(image["width"]), int(image["height"]))
            if coordinates is None:
                reasons.append("invalid_bbox")
            accepted = not reasons
            if accepted:
                lines.append("0 " + " ".join(f"{value:.6f}" for value in coordinates or []))
                accepted_count += 1
            else:
                rejected_count += 1
            decisions.append({
                "accepted": accepted,
                "reasons": reasons,
                "bbox": annotation["bbox"],
                "source_prompts": annotation.get("source_prompts", []),
                **metrics,
            })
        (labels_dir / f"{source.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n"
        )
        audit_images.append({
            "image": source.name,
            "label": f"{source.stem}.txt",
            "accepted": len(lines),
            "candidates": len(decisions),
            "decisions": decisions,
        })

    if len(images) != len(source_files):
        raise SystemExit(
            f"image count mismatch: COCO has {len(images)}, local directory has {len(source_files)}"
        )
    audit = {
        "source_coco": str(args.coco.resolve()),
        "images_dir": str(args.images_dir.resolve()),
        "labels_dir": str(labels_dir.resolve()),
        "thresholds": {
            "min_score": args.min_score,
            "min_fill_ratio": args.min_fill_ratio,
            "min_prompt_votes": args.min_prompt_votes,
            "min_relative_area": args.min_relative_area,
        },
        "summary": {
            "images": len(images),
            "accepted_annotations": accepted_count,
            "rejected_annotations": rejected_count,
            "positive_images": sum(1 for item in audit_images if item["accepted"]),
        },
        "images": audit_images,
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    print(json.dumps(audit["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
