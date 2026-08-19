"""Lọc ảnh lỗi/trùng và chuẩn hóa ảnh gốc sang JPEG."""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from fire_detection.utils.console import configure_utf8_output


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class PreprocessStats:
    total: int = 0
    corrupt: int = 0
    duplicate: int = 0
    kept: int = 0


def image_content_hash(image) -> str:
    """Tạo hash nội dung ảnh đã decode để phát hiện bản sao chính xác."""

    digest = hashlib.sha256()
    digest.update(str(image.shape).encode("ascii"))
    digest.update(str(image.dtype).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def resize_max_side(image, max_side: int):
    """Thu nhỏ ảnh vượt giới hạn và giữ nguyên tỷ lệ; không phóng lớn ảnh nhỏ."""

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / longest
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, target, interpolation=cv2.INTER_AREA)


def preprocess_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    max_side: int = 1280,
    jpeg_quality: int = 95,
) -> PreprocessStats:
    """Đọc ảnh đệ quy, bỏ ảnh lỗi/trùng và ghi tập JPEG đã chuẩn hóa."""

    source = Path(input_dir)
    destination = Path(output_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {source}")
    if max_side <= 0:
        raise ValueError("max_side phải dương")
    if not 0 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality phải thuộc [0, 100]")

    source_resolved = source.resolve()
    destination_resolved = destination.resolve()
    if source_resolved == destination_resolved or source_resolved in destination_resolved.parents:
        raise ValueError("Thư mục đầu ra phải nằm ngoài thư mục đầu vào")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"Thư mục đầu ra phải rỗng để tránh lẫn dữ liệu từ lần chạy trước: {destination}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    stats = PreprocessStats()
    manifest_rows: list[dict[str, str]] = []

    candidates = sorted(
        path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )
    for image_path in candidates:
        stats.total += 1
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            stats.corrupt += 1
            continue

        content_hash = image_content_hash(image)
        if content_hash in seen_hashes:
            stats.duplicate += 1
            continue
        seen_hashes.add(content_hash)

        image = resize_max_side(image, max_side)
        output_path = destination / f"image_{stats.kept:06d}.jpg"
        written = cv2.imwrite(
            str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        )
        if not written:
            raise OSError(f"Không thể ghi ảnh: {output_path}")
        manifest_rows.append(
            {
                "source": image_path.relative_to(source).as_posix(),
                "output": output_path.name,
                "sha256_decoded": content_hash,
            }
        )
        stats.kept += 1

    manifest = destination / "preprocess_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source", "output", "sha256_decoded"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    return stats


def main() -> None:
    configure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw", help="Thư mục ảnh gốc")
    parser.add_argument("--output", default="data/interim/clean_images", help="Thư mục ảnh sạch")
    parser.add_argument("--max-side", type=int, default=1280)
    args = parser.parse_args()
    stats = preprocess_directory(args.input, args.output, max_side=args.max_side)
    print(asdict(stats))


if __name__ == "__main__":
    main()
