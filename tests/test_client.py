from pathlib import Path

import pytest

from client.batch_upload import (
    ClientError,
    apply_run_config,
    build_task,
    discover_images,
    make_parser,
    main,
    validate_args,
)


def test_discovers_supported_images_and_deduplicates(tmp_path: Path):
    first = tmp_path / "a.jpg"
    second = tmp_path / "nested" / "b.png"
    ignored = tmp_path / "notes.txt"
    second.parent.mkdir()
    first.write_bytes(b"jpg")
    second.write_bytes(b"png")
    ignored.write_text("ignored", encoding="utf-8")

    images = discover_images([str(tmp_path), str(first)], recursive=True)

    assert images == [first.resolve(), second.resolve()]


def test_builds_multiple_prompts_for_one_label():
    parser = make_parser()
    args = parser.parse_args(
        [
            "image.jpg",
            "--server",
            "http://example.test:8000",
            "--prompt",
            "person=person",
            "--prompt",
            "person=sleeping person",
            "--prompt",
            "chair=chair",
        ]
    )
    validate_args(args)
    task = build_task(args)

    assert task["prediction"]["imgsz"] == 644
    assert task["prompt_groups"][0]["prompts"] == ["person", "sleeping person"]
    assert task["prompt_groups"][1]["class_id"] == 1


def test_rejects_non_stride_aligned_imgsz():
    parser = make_parser()
    args = parser.parse_args(
        [
            "image.jpg",
            "--server",
            "http://example.test:8000",
            "--prompt",
            "person=person",
            "--imgsz",
            "640",
        ]
    )
    with pytest.raises(ClientError, match="divisible by 14"):
        validate_args(args)


def test_toml_run_config(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "sample.jpg").write_bytes(b"jpg")
    config = tmp_path / "batch.toml"
    config.write_text(
        """
[client]
server = "http://example.test:8000"
inputs = ["images"]
output_dir = "output"
batch_size = 4

[task]
task_type = "segment"
output_format = "yolo"
aggregation = "deduplicate"
merge_iou = 0.7

[task.prediction]
conf = 0.5
iou = 0.7
imgsz = 644
max_det = 100
retina_masks = true

[task.labels]
person = ["person", "sleeping person"]
chair = "chair"
""".strip(),
        encoding="utf-8",
    )
    args = make_parser().parse_args(["--config", str(config)])

    apply_run_config(args)
    validate_args(args)
    task = build_task(args)

    assert args.inputs == [str(tmp_path / "images")]
    assert args.output_dir == tmp_path / "output"
    assert args.batch_size == 4
    assert task["prompt_groups"][0]["prompts"] == ["person", "sleeping person"]
    assert task["prompt_groups"][1]["class_id"] == 1


def test_command_line_dry_run_overrides_config(tmp_path: Path, capsys):
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"jpg")
    config = tmp_path / "batch.toml"
    config.write_text(
        f'''\
[client]
server = "http://example.test:8000"
inputs = ["{image.name}"]
dry_run = false

[task]
task_type = "detect"
output_format = "yolo"

[task.labels]
person = "person"
''',
        encoding="utf-8",
    )

    assert main(["--config", str(config), "--dry-run"]) == 0
    assert "sample.jpg" in capsys.readouterr().out
