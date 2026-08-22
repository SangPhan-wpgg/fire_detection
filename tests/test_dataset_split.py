import csv
from pathlib import Path

import pytest

from fire_detection.data.split import split_yolo_dataset


def _write_sample(images: Path, labels: Path, name: str, label: str = "") -> None:
    (images / f"{name}.jpg").write_bytes(b"image")
    (labels / f"{name}.txt").write_text(label, encoding="utf-8")


def test_grouped_split_keeps_scene_together(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    output = tmp_path / "output"
    images.mkdir()
    labels.mkdir()
    names = ["scene_a_1", "scene_a_2", "scene_b_1", "scene_c_1", "scene_d_1"]
    for name in names:
        _write_sample(images, labels, name, "0 0.5 0.5 0.2 0.2\n")

    groups = tmp_path / "groups.csv"
    groups.write_text(
        "image,group\n"
        "scene_a_1.jpg,scene_a\n"
        "scene_a_2.jpg,scene_a\n"
        "scene_b_1.jpg,scene_b\n"
        "scene_c_1.jpg,scene_c\n"
        "scene_d_1.jpg,scene_d\n",
        encoding="utf-8",
    )
    manifest = split_yolo_dataset(
        images,
        labels,
        output,
        train_ratio=0.5,
        valid_ratio=0.25,
        group_manifest=groups,
    )

    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    scene_a_splits = {row["split"] for row in rows if row["group"] == "scene_a"}
    assert len(scene_a_splits) == 1
    assert {row["image"] for row in rows} == {f"{name}.jpg" for name in names}


def test_split_rejects_missing_label(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (images / "missing.jpg").write_bytes(b"image")

    with pytest.raises(ValueError, match="Thiếu nhãn"):
        split_yolo_dataset(images, labels, tmp_path / "output")


def test_split_rejects_nonempty_output(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    output = tmp_path / "output"
    images.mkdir()
    labels.mkdir()
    output.mkdir()
    (output / "stale.txt").write_text("stale", encoding="utf-8")
    _write_sample(images, labels, "sample")

    with pytest.raises(FileExistsError):
        split_yolo_dataset(images, labels, output)
