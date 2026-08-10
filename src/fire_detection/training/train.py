"""Huấn luyện YOLO từ một cấu hình YAML có thể truy vết."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fire_detection.config import load_yaml, resolve_project_path
from fire_detection.models.yolov8 import build_yolov8
from fire_detection.utils.console import configure_utf8_output
from fire_detection.utils.reproducibility import seed_everything


def train_from_config(config_path: str | Path) -> Any:
    config = load_yaml(config_path)
    if config.get("task") not in {"detect", "segment"}:
        raise ValueError("Script hiện hỗ trợ task YOLO 'detect' và 'segment'")

    train_args = dict(config.get("train", {}))
    seed = int(train_args.get("seed", 42))
    seed_everything(seed)
    data_path = resolve_project_path(config["data"])
    if not data_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy cấu hình dữ liệu: {data_path}")

    if "project" in train_args:
        train_args["project"] = str(resolve_project_path(train_args["project"]))
    model = build_yolov8(str(config["model"]))
    return model.train(data=str(data_path), **train_args)


def main() -> None:
    configure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/models/yolov8n.yaml")
    args = parser.parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()
