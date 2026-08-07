import pytest

from fire_detection.evaluation.detection_metrics import f1_score, precision, recall


def test_detection_metrics() -> None:
    assert precision(8, 2) == 0.8
    assert recall(8, 2) == 0.8
    assert f1_score(0.8, 0.8) == pytest.approx(0.8)


def test_detection_metrics_zero_denominator() -> None:
    assert precision(0, 0) == 0.0
    assert recall(0, 0) == 0.0
    assert f1_score(0.0, 0.0) == 0.0
