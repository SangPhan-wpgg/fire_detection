"""Chia dữ liệu YOLO theo nhóm với seed cố định và lưu manifest."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

from fire_detection.data.validation import validate_detection_dataset
from fire_detection.utils.console import configure_utf8_output


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _load_group_manifest(path: str | Path, image_names: set[str]) -> dict[str, str]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy group manifest: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"image", "group"}.issubset(reader.fieldnames):
            raise ValueError("Group manifest phải có hai cột 'image' và 'group'")
        mapping = {
            str(row["image"]).strip(): str(row["group"]).strip()
            for row in reader
            if row.get("image") and row.get("group")
        }
    missing = sorted(image_names - mapping.keys())
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Group manifest thiếu {len(missing)} ảnh, ví dụ: {preview}")
    return mapping


def split_yolo_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    output_dir: str | Path,
    *,
    train_ratio: float = 0.75,
    valid_ratio: float = 0.15,
    seed: int = 42,
    class_count: int = 1,
    group_manifest: str | Path | None = None,
    allow_missing_labels: bool = False,
) -> Path:
    """Chia cặp image-label trước augmentation và giữ mỗi scene trong một split.

    ``group_manifest`` là CSV gồm hai cột ``image,group``. Các ảnh cùng group
    luôn được đưa vào cùng một split. Nếu không truyền manifest, mỗi ảnh là một
    group độc lập.
    """

    if train_ratio <= 0 or valid_ratio <= 0 or train_ratio + valid_ratio >= 1:
        raise ValueError("Tỷ lệ train/valid phải dương và có tổng nhỏ hơn 1")
    if class_count <= 0:
        raise ValueError("class_count phải dương")

    image_root = Path(images_dir)
    label_root = Path(labels_dir)
    destination = Path(output_dir)
    if not image_root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {image_root}")
    if not label_root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục nhãn: {label_root}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"Thư mục đầu ra phải rỗng để tránh trộn các lần chia tập: {destination}"
        )

    images = sorted(
        path for path in image_root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"Không có ảnh trong {image_root}")
    stems = [path.stem for path in images]
    if len(stems) != len(set(stems)):
        raise ValueError("Tên ảnh bị trùng stem giữa các phần mở rộng")

    issues = validate_detection_dataset(
        image_root,
        label_root,
        class_count=class_count,
        allow_missing_labels=allow_missing_labels,
    )
    if issues:
        preview = "; ".join(
            f"{issue.file}:{issue.line} {issue.message}" for issue in issues[:5]
        )
        raise ValueError(f"Dataset có {len(issues)} lỗi nhãn. {preview}")

    image_names = {path.name for path in images}
    group_mapping = (
        _load_group_manifest(group_manifest, image_names)
        if group_manifest is not None
        else {path.name: path.stem for path in images}
    )
    grouped_images: dict[str, list[Path]] = defaultdict(list)
    for image_path in images:
        grouped_images[group_mapping[image_path.name]].append(image_path)

    group_ids = sorted(grouped_images)
    random.Random(seed).shuffle(group_ids)
    train_target = int(len(images) * train_ratio)
    valid_target = int(len(images) * valid_ratio)
    assignments: dict[str, list[Path]] = {"train": [], "valid": [], "test": []}
    assigned_groups: dict[str, str] = {}
    for group_id in group_ids:
        if len(assignments["train"]) < train_target:
            split_name = "train"
        elif len(assignments["valid"]) < valid_target:
            split_name = "valid"
        else:
            split_name = "test"
        assignments[split_name].extend(grouped_images[group_id])
        assigned_groups[group_id] = split_name

    destination.mkdir(parents=True, exist_ok=True)
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
            group_id = group_mapping[image_path.name]
            manifest_rows.append(
                {
                    "image": image_path.name,
                    "label": target_label.name,
                    "split": assigned_groups[group_id],
                    "group": group_id,
                }
            )

    manifest = destination / "split_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "label", "split", "group"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest


def main() -> None:
    configure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-count", type=int, default=1)
    parser.add_argument("--group-manifest")
    parser.add_argument("--allow-missing-labels", action="store_true")
    args = parser.parse_args()
    manifest = split_yolo_dataset(
        args.images,
        args.labels,
        args.output,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        class_count=args.class_count,
        group_manifest=args.group_manifest,
        allow_missing_labels=args.allow_missing_labels,
    )
    print(manifest)


if __name__ == "__main__":
    main()
