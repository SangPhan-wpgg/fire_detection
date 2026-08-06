"""Kiểm tra cú pháp nhãn YOLO detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
        if not all(0 <= value <= 1 for value in (x_center, y_center, width, height)):
            issues.append(LabelIssue(str(label_path), line_number, "Tọa độ phải thuộc [0, 1]"))
        if width <= 0 or height <= 0:
            issues.append(LabelIssue(str(label_path), line_number, "Kích thước hộp phải dương"))
    return issues
