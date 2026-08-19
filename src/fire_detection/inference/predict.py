"""Suy luận bằng checkpoint YOLO trên ảnh, thư mục ảnh hoặc video local."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fire_detection.config import resolve_project_path
from fire_detection.models.yolov8 import build_yolov8
from fire_detection.utils.console import configure_utf8_output


def predict(
    checkpoint: str | Path,
    source: str | Path,
    *,
    confidence: float = 0.25,
    image_size: int = 640,
    project: str | Path = "artifacts/predictions",
    name: str = "predict",
) -> Any:
    """Chạy dự đoán với đầu vào và checkpoint local đã được kiểm tra."""

    if not 0 <= confidence <= 1:
        raise ValueError("confidence phải thuộc [0, 1]")
    if image_size <= 0:
        raise ValueError("image_size phải dương")

    checkpoint_path = resolve_project_path(checkpoint)
    source_path = resolve_project_path(source)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Không tìm thấy nguồn suy luận: {source_path}")

    output_root = resolve_project_path(project)
    model = build_yolov8(str(checkpoint_path))
    return model.predict(
        source=str(source_path),
        conf=confidence,
        imgsz=image_size,
        project=str(output_root),
        name=name,
        save=True,
    )


def main() -> None:
    configure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default="artifacts/predictions")
    parser.add_argument("--name", default="predict")
    args = parser.parse_args()
    predict(
        args.checkpoint,
        args.source,
        confidence=args.conf,
        image_size=args.imgsz,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
