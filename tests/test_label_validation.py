from pathlib import Path

from fire_detection.data.validation import validate_detection_label


FIXTURES = Path(__file__).parent / "fixtures"


def test_valid_detection_label() -> None:
    label = FIXTURES / "valid_label.txt"
    assert validate_detection_label(label) == []


def test_invalid_detection_label() -> None:
    label = FIXTURES / "invalid_label.txt"
    issues = validate_detection_label(label)
    assert {issue.message for issue in issues} == {
        "class_id ngoài phạm vi",
        "Tọa độ phải thuộc [0, 1]",
        "Kích thước hộp phải dương",
    }
