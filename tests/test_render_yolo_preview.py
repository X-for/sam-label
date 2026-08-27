from client.render_yolo_preview import positive_label_paths


def test_positive_label_paths_excludes_classes_and_empty_labels(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "classes.txt").write_text("car\n", encoding="utf-8")
    (labels_dir / "CLASSES.TXT").write_text("truck\n", encoding="utf-8")
    (labels_dir / "empty.txt").write_text("", encoding="utf-8")
    (labels_dir / "image_002.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (labels_dir / "image_001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    assert [path.name for path in positive_label_paths(labels_dir)] == [
        "image_001.txt",
        "image_002.txt",
    ]
