from pathlib import Path

from fire_detection.config import PROJECT_ROOT, resolve_project_path, resolve_yolo_split_paths


def test_project_root_points_to_repository() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert resolve_project_path("configs/data/detection.yaml") == (
        PROJECT_ROOT / "configs/data/detection.yaml"
    ).resolve()


def test_absolute_path_is_preserved(tmp_path: Path) -> None:
    assert resolve_project_path(tmp_path) == tmp_path.resolve()


def test_detection_splits_resolve_independently_from_cwd() -> None:
    splits = resolve_yolo_split_paths("configs/data/detection.yaml")
    assert splits == {
        "train": (PROJECT_ROOT / "data/processed/detection/train/images").resolve(),
        "val": (PROJECT_ROOT / "data/processed/detection/valid/images").resolve(),
        "test": (PROJECT_ROOT / "data/processed/detection/test/images").resolve(),
    }
