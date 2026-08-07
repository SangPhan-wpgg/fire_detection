"""Đọc cấu hình và chuẩn hóa đường dẫn của dự án."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    """Trả đường dẫn tuyệt đối; đường dẫn tương đối được tính từ project root."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Đọc một tệp YAML và yêu cầu nút gốc là mapping."""

    config_path = resolve_project_path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Cấu hình phải là mapping YAML: {config_path}")
    return config
