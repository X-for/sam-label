#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply an OCR verification report to YOLO labels.")
    parser.add_argument("report", type=Path)
    parser.add_argument("datasets", type=Path)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    kept_by_dataset: Counter[str] = Counter()
    positive_by_dataset: Counter[str] = Counter()
    for image in report["images"]:
        label_path = args.datasets / image["dataset"] / "labels" / image["label"]
        original = label_path.read_text(encoding="utf-8").splitlines()
        expected = [candidate["yolo"] for candidate in image["candidates"]]
        if original != expected:
            raise SystemExit(f"label changed since OCR report was generated: {label_path}")
        kept = [candidate["yolo"] for candidate in image["candidates"] if candidate["matched"]]
        label_path.write_text(
            "\n".join(kept) + ("\n" if kept else ""), encoding="utf-8", newline="\n"
        )
        kept_by_dataset[image["dataset"]] += len(kept)
        if kept:
            positive_by_dataset[image["dataset"]] += 1

    summary = {
        dataset: {
            "annotations": kept_by_dataset[dataset],
            "positive_images": positive_by_dataset[dataset],
        }
        for dataset in sorted(path.name for path in args.datasets.iterdir() if path.is_dir())
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
