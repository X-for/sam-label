#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "onnxruntime>=1.22,<2",
#   "pillow>=11,<13",
#   "rapidocr>=3.9,<4",
# ]
# ///
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from rapidocr import RapidOCR


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR-check YOLO candidates for the Chinese character 爆.")
    parser.add_argument("datasets", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rotations", type=int, nargs="+", default=(0, 90, 180, 270))
    parser.add_argument("--padding", type=float, default=0.12)
    return parser.parse_args()


def image_map(images_dir: Path) -> dict[str, Path]:
    result = {
        path.stem.casefold(): path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    return result


def crop_box(image: Image.Image, values: list[str], padding: float) -> Image.Image:
    _, cx, cy, width, height = values
    cx, cy, width, height = map(float, (cx, cy, width, height))
    pad_x = width * padding
    pad_y = height * padding
    x1 = max(0, round((cx - width / 2 - pad_x) * image.width))
    y1 = max(0, round((cy - height / 2 - pad_y) * image.height))
    x2 = min(image.width, round((cx + width / 2 + pad_x) * image.width))
    y2 = min(image.height, round((cy + height / 2 + pad_y) * image.height))
    return image.crop((x1, y1, x2, y2))


def recognize(engine: RapidOCR, crop: Image.Image, rotations: list[int]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for rotation in rotations:
        rotated = crop.rotate(rotation, expand=True)
        pixels = np.asarray(rotated)
        output = engine(pixels)
        texts = list(output.txts or ())
        scores = [round(float(score), 5) for score in (output.scores or ())]
        direct = engine(pixels, use_det=False, use_cls=False, use_rec=True)
        raw_direct_texts = direct.txts or ()
        direct_text = (
            raw_direct_texts
            if isinstance(raw_direct_texts, str)
            else " ".join(str(text) for text in raw_direct_texts)
        )
        raw_direct_scores = direct.scores or ()
        if isinstance(raw_direct_scores, (float, int)):
            direct_score = round(float(raw_direct_scores), 5)
        else:
            direct_score = round(max((float(score) for score in raw_direct_scores), default=0.0), 5)
        results.append({
            "rotation": rotation,
            "texts": texts,
            "scores": scores,
            "direct_text": direct_text,
            "direct_score": direct_score,
        })
    return results


def main() -> int:
    args = parse_args()
    candidates: list[tuple[Path, Path]] = []
    for dataset_dir in sorted(path for path in args.datasets.iterdir() if path.is_dir()):
        labels_dir = dataset_dir / "labels"
        images_dir = dataset_dir / "images"
        if not labels_dir.is_dir() or not images_dir.is_dir():
            continue
        images = image_map(images_dir)
        for label_path in sorted(labels_dir.glob("*.txt"), key=lambda path: path.name.casefold()):
            if label_path.stat().st_size:
                image_path = images.get(label_path.stem.casefold())
                if image_path is None:
                    raise SystemExit(f"image missing for {label_path}")
                candidates.append((image_path, label_path))
    if args.limit:
        step = max(1, len(candidates) // args.limit)
        candidates = candidates[::step][: args.limit]

    engine = RapidOCR()
    report: list[dict[str, Any]] = []
    total = sum(len(path.read_text(encoding="utf-8").splitlines()) for _, path in candidates)
    completed = 0
    for image_path, label_path in candidates:
        image = Image.open(image_path).convert("RGB")
        entries = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            crop = crop_box(image, values, args.padding)
            ocr = recognize(engine, crop, args.rotations)
            texts = [text for item in ocr for text in [*item["texts"], item["direct_text"]]]
            entries.append({"yolo": line, "matched": any("爆" in text for text in texts), "ocr": ocr})
            completed += 1
            if completed % 20 == 0 or completed == total:
                print(f"OCR {completed}/{total}", flush=True)
        report.append({
            "dataset": image_path.parent.parent.name,
            "image": image_path.name,
            "label": label_path.name,
            "candidates": entries,
        })
    summary = {
        "images": len(report),
        "candidates": total,
        "matched": sum(
            1 for item in report for candidate in item["candidates"] if candidate["matched"]
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({"summary": summary, "images": report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
