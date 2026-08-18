"""Đọc cấu hình và chuẩn hóa đường dẫn của dự án."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    """Trả đường dẫn tuyệt đối; đường dẫn tương đối được tính từ project root."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Đọc một tệp YAML và yêu cầu nút gốc là mapping."""

    config_path = resolve_project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy cấu hình: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Cấu hình phải là mapping YAML: {config_path}")
    return config


def resolve_yolo_split_paths(path: str | Path) -> dict[str, Path]:
    """Giải quyết các split local trong data YAML theo chính vị trí tệp YAML."""

    config_path = resolve_project_path(path)
    config = load_yaml(config_path)
    declared_root = config.get("path")
    if declared_root is None:
        dataset_root = config_path.parent
    else:
        root_candidate = Path(str(declared_root)).expanduser()
        dataset_root = (
            root_candidate.resolve()
            if root_candidate.is_absolute()
            else (PROJECT_ROOT / root_candidate).resolve()
        )

    splits: dict[str, Path] = {}
    for split_name in ("train", "val", "test"):
        split_value = config.get(split_name)
        if split_value is None:
            continue
        if not isinstance(split_value, str):
            raise ValueError(f"Split '{split_name}' phải là một đường dẫn chuỗi")
        candidate = Path(split_value).expanduser()
        splits[split_name] = (
            candidate.resolve() if candidate.is_absolute() else (dataset_root / candidate).resolve()
        )
    return splits
