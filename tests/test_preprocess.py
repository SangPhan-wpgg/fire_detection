import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from fire_detection.data.preprocess import preprocess_directory


def test_preprocess_deduplicates_and_writes_manifest(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "clean"
    source.mkdir()
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    assert cv2.imwrite(str(source / "first.png"), image)
    assert cv2.imwrite(str(source / "duplicate.png"), image)

    stats = preprocess_directory(source, output)

    assert stats.total == 2
    assert stats.kept == 1
    assert stats.duplicate == 1
    with (output / "preprocess_manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["output"] == "image_000000.jpg"


def test_preprocess_rejects_nonempty_output(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "clean"
    source.mkdir()
    output.mkdir()
    (output / "stale.jpg").write_bytes(b"stale")

    with pytest.raises(FileExistsError):
        preprocess_directory(source, output)
