"""Kiểm tra cú pháp nhãn YOLO detection."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from fire_detection.utils.console import configure_utf8_output


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class LabelIssue:
    file: str
    line: int
    message: str


def validate_detection_label(path: str | Path, class_count: int = 1) -> list[LabelIssue]:
    """Trả danh sách lỗi class và tọa độ của một tệp nhãn YOLO."""

    label_path = Path(path)
    issues: list[LabelIssue] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = raw_line.split()
        if len(parts) != 5:
            issues.append(LabelIssue(str(label_path), line_number, "Nhãn phải có 5 giá trị"))
            continue
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            issues.append(LabelIssue(str(label_path), line_number, "Giá trị nhãn không hợp lệ"))
            continue
        if not 0 <= class_id < class_count:
            issues.append(LabelIssue(str(label_path), line_number, "class_id ngoài phạm vi"))
        if not all(math.isfinite(value) for value in (x_center, y_center, width, height)):
            issues.append(LabelIssue(str(label_path), line_number, "Tọa độ phải là số hữu hạn"))
            continue
        if not all(0 <= value <= 1 for value in (x_center, y_center, width, height)):
            issues.append(LabelIssue(str(label_path), line_number, "Tọa độ phải thuộc [0, 1]"))
        if width <= 0 or height <= 0:
            issues.append(LabelIssue(str(label_path), line_number, "Kích thước hộp phải dương"))
        elif all(0 <= value <= 1 for value in (x_center, y_center, width, height)) and (
            x_center - width / 2 < 0
            or x_center + width / 2 > 1
            or y_center - height / 2 < 0
            or y_center + height / 2 > 1
        ):
            issues.append(LabelIssue(str(label_path), line_number, "Hộp giới hạn vượt biên ảnh"))
    return issues


def validate_detection_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    *,
    class_count: int = 1,
    allow_missing_labels: bool = False,
) -> list[LabelIssue]:
    """Kiểm tra quan hệ image-label và nội dung toàn bộ nhãn detection."""

    image_root = Path(images_dir)
    label_root = Path(labels_dir)
    if not image_root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {image_root}")
    if not label_root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục nhãn: {label_root}")
    if class_count <= 0:
        raise ValueError("class_count phải dương")

    issues: list[LabelIssue] = []
    image_stems = {
        path.stem
        for path in image_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    label_files = [
        path
        for path in label_root.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    ]
    label_paths = {path.stem: path for path in label_files}
    if len(label_paths) != len(label_files):
        issues.append(LabelIssue(str(label_root), 0, "Tên label bị trùng stem"))

    for stem in sorted(image_stems):
        label_path = label_paths.get(stem)
        if label_path is None:
            if not allow_missing_labels:
                issues.append(LabelIssue(str(label_root / f"{stem}.txt"), 0, "Thiếu nhãn cho ảnh"))
            continue
        issues.extend(validate_detection_label(label_path, class_count=class_count))

    for stem, label_path in sorted(label_paths.items()):
        if stem not in image_stems:
            issues.append(LabelIssue(str(label_path), 0, "Nhãn không có ảnh tương ứng"))
    return issues


def main() -> None:
    configure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--class-count", type=int, default=1)
    parser.add_argument("--allow-missing-labels", action="store_true")
    args = parser.parse_args()
    issues = validate_detection_dataset(
        args.images,
        args.labels,
        class_count=args.class_count,
        allow_missing_labels=args.allow_missing_labels,
    )
    for issue in issues:
        print(f"{issue.file}:{issue.line}: {issue.message}")
    if issues:
        raise SystemExit(1)
    print("Dataset detection hợp lệ")


if __name__ == "__main__":
    main()
