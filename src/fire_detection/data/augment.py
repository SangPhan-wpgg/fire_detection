"""Tăng cường ảnh nền của tập train; không dùng cho validation/test."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def geometric_augmentation(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = image.copy()
    if rng.random() < 0.5:
        result = cv2.flip(result, 1)
    if rng.random() < 0.5:
        result = cv2.flip(result, 0)
    height, width = result.shape[:2]
    angle = float(rng.uniform(-180, 180))
    matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
    return cv2.warpAffine(result, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)


def color_augmentation(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = image.astype(np.float32)
    result += float(rng.uniform(-30, 30))
    result *= float(rng.uniform(0.8, 1.3))
    result = np.clip(result, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= float(rng.uniform(0.85, 1.2))
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def noise_blur_augmentation(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, 15, image.shape)
    result = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        result = cv2.GaussianBlur(result, (5, 5), 0)
    if rng.random() < 0.3:
        result = cv2.blur(result, (3, 3))
    return result


def fog_augmentation(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = image.shape[:2]
    fog = np.zeros_like(image)
    minimum = max(20, min(height, width) // 8)
    maximum = max(minimum + 1, min(height, width) // 2)
    for _ in range(int(rng.integers(1, 4))):
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        radius = int(rng.integers(minimum, maximum))
        cv2.circle(fog, center, radius, (255, 255, 255), -1)
    kernel = min(151, (min(height, width) // 2) * 2 - 1)
    kernel = max(3, kernel if kernel % 2 == 1 else kernel - 1)
    fog = cv2.GaussianBlur(fog, (kernel, kernel), 0)
    alpha = float(rng.uniform(0.1, 0.3))
    return cv2.addWeighted(image, 1 - alpha, fog, alpha, 0)


def augment_negative_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
) -> int:
    """Sinh bốn biến thể cho mỗi ảnh negative thuộc train và trả số ảnh đã ghi."""

    source = Path(input_dir)
    destination = Path(output_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    transforms = {
        "geom": geometric_augmentation,
        "color": color_augmentation,
        "noise": noise_blur_augmentation,
        "fog": fog_augmentation,
    }
    count = 0
    for image_path in sorted(source.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        for prefix, transform in transforms.items():
            output_path = destination / f"{prefix}_{image_path.stem}.jpg"
            if not cv2.imwrite(str(output_path), transform(image, rng)):
                raise OSError(f"Không thể ghi ảnh: {output_path}")
            count += 1
    return count
