#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a contact sheet of YOLO detection labels.")
    parser.add_argument("datasets", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples-per-set", type=int, default=4)
    return parser.parse_args()


def evenly_spaced(items: list[Path], count: int) -> list[Path]:
    if len(items) <= count:
        return items
    return [items[round(index * (len(items) - 1) / (count - 1))] for index in range(count)]


def find_image(images_dir: Path, stem: str) -> Path:
    matches = [path for suffix in IMAGE_SUFFIXES for path in images_dir.glob(f"{stem}{suffix}")]
    if len(matches) != 1:
        raise SystemExit(f"expected one image for label {stem}, found {len(matches)}")
    return matches[0]


def draw_preview(image_path: Path, label_path: Path, caption: str) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5:
            continue
        _, cx, cy, width, height = values
        cx, cy, width, height = map(float, (cx, cy, width, height))
        x1 = (cx - width / 2) * image.width
        y1 = (cy - height / 2) * image.height
        x2 = (cx + width / 2) * image.width
        y2 = (cy + height / 2) * image.height
        stroke = max(3, round(min(image.size) / 250))
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 80), width=stroke)
    image.thumbnail((470, 290), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (480, 320), "white")
    tile.paste(image, ((480 - image.width) // 2, 24 + (290 - image.height) // 2))
    ImageDraw.Draw(tile).text((8, 5), caption, fill="black")
    return ImageOps.expand(tile, border=1, fill=(180, 180, 180))


def main() -> int:
    args = parse_args()
    tiles: list[Image.Image] = []
    for dataset_dir in sorted(path for path in args.datasets.iterdir() if path.is_dir()):
        labels_dir = dataset_dir / "labels"
        images_dir = dataset_dir / "images"
        if not labels_dir.is_dir() or not images_dir.is_dir():
            continue
        positives = sorted(
            (path for path in labels_dir.glob("*.txt") if path.stat().st_size),
            key=lambda path: path.name.casefold(),
        )
        for index, label_path in enumerate(evenly_spaced(positives, args.samples_per_set), start=1):
            image_path = find_image(images_dir, label_path.stem)
            tiles.append(draw_preview(image_path, label_path, f"{dataset_dir.name} sample {index}"))

    columns = args.samples_per_set
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 482, rows * 322), (235, 235, 235))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 482, (index // columns) * 322))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=92)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
