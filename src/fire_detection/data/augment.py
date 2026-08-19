"""Tăng cường ảnh nền của tập train; không dùng cho validation/test."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from fire_detection.utils.console import configure_utf8_output


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
    labels_output_dir: str | Path | None = None,
    require_train_path: bool = True,
) -> int:
    """Sinh bốn biến thể cho ảnh negative thuộc train và tạo nhãn rỗng tùy chọn."""

    source = Path(input_dir)
    destination = Path(output_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {source}")
    if require_train_path and "train" not in {part.lower() for part in source.parts}:
        raise ValueError("Chỉ được augmentation ảnh thuộc đường dẫn train")
    if source.resolve() == destination.resolve():
        raise ValueError("Thư mục đầu vào và đầu ra phải khác nhau")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Thư mục augmentation đầu ra phải rỗng: {destination}")
    label_destination = Path(labels_output_dir) if labels_output_dir is not None else None
    if label_destination is not None and label_destination.resolve() == destination.resolve():
        raise ValueError("Thư mục ảnh và nhãn augmentation phải khác nhau")
    if label_destination is not None and label_destination.exists() and any(label_destination.iterdir()):
        raise FileExistsError(f"Thư mục nhãn augmentation phải rỗng: {label_destination}")
    destination.mkdir(parents=True, exist_ok=True)
    if label_destination is not None:
        label_destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    transforms = {
        "geom": geometric_augmentation,
        "color": color_augmentation,
        "noise": noise_blur_augmentation,
        "fog": fog_augmentation,
    }
    count = 0
    image_paths = [
        path
        for path in sorted(source.iterdir())
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    stems = [path.stem for path in image_paths]
    if len(stems) != len(set(stems)):
        raise ValueError("Tên ảnh negative bị trùng stem giữa các phần mở rộng")
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        for prefix, transform in transforms.items():
            output_path = destination / f"{prefix}_{image_path.stem}.jpg"
            if not cv2.imwrite(str(output_path), transform(image, rng)):
                raise OSError(f"Không thể ghi ảnh: {output_path}")
            if label_destination is not None:
                (label_destination / f"{output_path.stem}.txt").touch()
            count += 1
    return count


def main() -> None:
    configure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Thư mục ảnh negative thuộc train")
    parser.add_argument("--output", required=True, help="Thư mục ảnh augmentation rỗng")
    parser.add_argument("--labels-output", help="Thư mục nhận các label rỗng tương ứng")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-non-train-path",
        action="store_true",
        help="Chỉ dùng khi đường dẫn không chứa tên 'train' nhưng dữ liệu đã được xác nhận là train",
    )
    args = parser.parse_args()
    count = augment_negative_directory(
        args.input,
        args.output,
        seed=args.seed,
        labels_output_dir=args.labels_output,
        require_train_path=not args.allow_non_train_path,
    )
    print({"augmented_images": count})


if __name__ == "__main__":
    main()
