"""Chia dữ liệu YOLO có seed cố định và lưu manifest."""

from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_yolo_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    output_dir: str | Path,
    *,
    train_ratio: float = 0.75,
    valid_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    """Chia cặp image-label; phải gọi trước khi tạo augmentation."""

    if train_ratio <= 0 or valid_ratio <= 0 or train_ratio + valid_ratio >= 1:
        raise ValueError("Tỷ lệ train/valid phải dương và có tổng nhỏ hơn 1")

    image_root = Path(images_dir)
    label_root = Path(labels_dir)
    destination = Path(output_dir)
    images = sorted(
        path for path in image_root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"Không có ảnh trong {image_root}")

    random.Random(seed).shuffle(images)
    train_end = int(len(images) * train_ratio)
    valid_end = train_end + int(len(images) * valid_ratio)
    assignments = {
        "train": images[:train_end],
        "valid": images[train_end:valid_end],
        "test": images[valid_end:],
    }

    manifest_rows: list[dict[str, str]] = []
    for split_name, split_images in assignments.items():
        target_images = destination / split_name / "images"
        target_labels = destination / split_name / "labels"
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)
        for image_path in split_images:
            label_path = label_root / f"{image_path.stem}.txt"
            shutil.copy2(image_path, target_images / image_path.name)
            target_label = target_labels / label_path.name
            if label_path.exists():
                shutil.copy2(label_path, target_label)
            else:
                target_label.touch()
            manifest_rows.append(
                {"image": image_path.name, "label": target_label.name, "split": split_name}
            )

    manifest = destination / "split_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "label", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest
