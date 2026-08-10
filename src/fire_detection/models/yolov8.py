"""Factory YOLOv8 với import Ultralytics trì hoãn."""

from __future__ import annotations


def build_yolov8(checkpoint: str = "yolov8n.pt"):
    from ultralytics import YOLO

    return YOLO(checkpoint)
