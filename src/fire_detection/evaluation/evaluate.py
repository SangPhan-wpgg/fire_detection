"""Đánh giá checkpoint YOLO trên split được khai báo trong cấu hình."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fire_detection.config import load_yaml, resolve_project_path
from fire_detection.models.yolov8 import build_yolov8
from fire_detection.utils.console import configure_utf8_output


def evaluate_from_config(config_path: str | Path, checkpoint: str | Path) -> Any:
    config = load_yaml(config_path)
    evaluation_args = dict(config.get("evaluation", {}))
    data_path = resolve_project_path(config["data"])
    model_path = resolve_project_path(checkpoint)
    if not data_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy cấu hình dữ liệu: {data_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {model_path}")
    if "project" in evaluation_args:
        evaluation_args["project"] = str(resolve_project_path(evaluation_args["project"]))
    model = build_yolov8(str(model_path))
    return model.val(data=str(data_path), **evaluation_args)


def main() -> None:
    configure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/models/yolov8n.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    evaluate_from_config(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
