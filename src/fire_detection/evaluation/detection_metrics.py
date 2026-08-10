"""Các metric phát hiện đơn giản dùng cho bảng tổng hợp."""

from __future__ import annotations


def precision(true_positive: int, false_positive: int) -> float:
    denominator = true_positive + false_positive
    return true_positive / denominator if denominator else 0.0


def recall(true_positive: int, false_negative: int) -> float:
    denominator = true_positive + false_negative
    return true_positive / denominator if denominator else 0.0


def f1_score(precision_value: float, recall_value: float) -> float:
    denominator = precision_value + recall_value
    return 2 * precision_value * recall_value / denominator if denominator else 0.0
