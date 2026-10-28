"""Chuẩn hóa ảnh EMNIST và ảnh ký tự do người dùng cung cấp."""

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from config import IMAGE_HEIGHT, IMAGE_WIDTH


def load_grayscale_image(image_path: Path) -> np.ndarray:
    """Đọc ảnh, ghép alpha lên nền trắng và trả mảng grayscale uint8."""
    try:
        with Image.open(image_path) as source:
            if "A" in source.getbands():
                rgba = source.convert("RGBA")
                white_background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                grayscale = Image.alpha_composite(white_background, rgba).convert("L")
            else:
                grayscale = source.convert("L")
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Không thể đọc ảnh ký tự '{image_path}': {error}") from error
    pixels = np.asarray(grayscale, dtype=np.uint8)
    if pixels.ndim != 2 or not pixels.size:
        raise ValueError("Ảnh đầu vào không có pixel.")
    return pixels


def orient_emnist_images(images: np.ndarray) -> np.ndarray:
    """Đổi trục ảnh IDX sang chiều canonical mà mô hình dùng."""
    if images.ndim != 3 or images.shape[1:] != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(
            f"Batch ảnh phải có dạng (N, {IMAGE_HEIGHT}, {IMAGE_WIDTH})."
        )
    return np.transpose(images, (0, 2, 1)).copy()


def grayscale_images_to_vectors(images: np.ndarray) -> np.ndarray:
    """Làm phẳng batch ảnh canonical 28x28 và scale cường độ về [0, 1]."""
    image_batch = np.asarray(images)
    if image_batch.ndim != 3 or image_batch.shape[1:] != (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    ):
        raise ValueError(
            f"Batch ảnh canonical phải có dạng (N, {IMAGE_HEIGHT}, {IMAGE_WIDTH})."
        )
    if image_batch.size and (
        not np.issubdtype(image_batch.dtype, np.number)
        or not np.isfinite(image_batch).all()
        or np.any(image_batch < 0)
        or np.any(image_batch > 255)
    ):
        raise ValueError("Cường độ pixel phải là số hữu hạn trong khoảng [0, 255].")
    return image_batch.reshape(image_batch.shape[0], -1).astype(np.float32) / 255.0


def dataset_images_to_vectors(images: np.ndarray) -> np.ndarray:
    """Định hướng EMNIST rồi dùng chung chuẩn hóa glyph với ảnh bên ngoài."""
    canonical_images = dataset_images_to_canonical(images)
    if not canonical_images:
        return np.empty((0, IMAGE_HEIGHT * IMAGE_WIDTH), dtype=np.float32)
    return np.stack([vector for _, vector in canonical_images])


def dataset_images_to_canonical(
    images: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Trả ảnh canonical và vector chuẩn hóa cho batch ảnh IDX."""
    oriented = orient_emnist_images(images)
    return [normalize_glyph(image) for image in oriented]


def preprocess_character_image(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Chuẩn hóa một ký tự để có cùng chiều và biểu diễn pixel như EMNIST."""
    return preprocess_character_image_array(load_grayscale_image(image_path))


def preprocess_character_image_array(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Chuẩn hóa crop bằng pipeline glyph dùng chung với EMNIST."""
    return normalize_glyph(image)


def normalize_glyph(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Chuẩn hóa polarity, nét, tỉ lệ và tâm thành glyph 28x28."""
    pixels = np.asarray(image)
    if (
        pixels.ndim != 2
        or pixels.size == 0
        or not np.issubdtype(pixels.dtype, np.number)
        or not np.isfinite(pixels).all()
        or np.any(pixels < 0)
        or np.any(pixels > 255)
    ):
        raise ValueError("Ảnh crop phải là grayscale không rỗng trong khoảng [0, 255].")
    pixels = pixels.astype(np.uint8)
    border = np.concatenate(
        (pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1])
    )

    if float(np.median(border)) > 127:
        pixels = 255 - pixels
        border = np.concatenate(
            (pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1])
        )

    # Loại nền đồng nhất/ám màu mà không tự tăng tương phản nét chữ.
    background_level = float(np.median(border))
    pixels = np.clip(
        pixels.astype(np.float32) - background_level, 0, 255
    ).astype(np.uint8)
    peak_intensity = int(pixels.max())
    if peak_intensity == 0:
        raise ValueError("Ảnh không có nét ký tự rõ; hãy kiểm tra ảnh đầu vào.")

    threshold = max(8, int(peak_intensity * 0.12))
    foreground = pixels > threshold
    rows, columns = np.where(foreground)
    if rows.size == 0:
        raise ValueError("Ảnh không có nét ký tự rõ; hãy kiểm tra ảnh đầu vào.")

    cropped = Image.fromarray(pixels).crop(
        (int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1)
    )
    # Giữ tỉ lệ khung bao; ký tự luôn nằm gọn trong vùng 20x20 của canvas.
    scale = min(20 / cropped.width, 20 / cropped.height)
    resized_size = (
        max(1, round(cropped.width * scale)),
        max(1, round(cropped.height * scale)),
    )
    cropped = cropped.resize(resized_size, Image.Resampling.LANCZOS)
    canvas = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT), color=0)
    offset = (
        (IMAGE_WIDTH - cropped.width) // 2,
        (IMAGE_HEIGHT - cropped.height) // 2,
    )
    canvas.paste(cropped, offset)
    canonical_image = np.asarray(canvas, dtype=np.uint8)
    canonical_image = np.rint(
        canonical_image.astype(np.float32) * (255.0 / canonical_image.max())
    ).clip(0, 255).astype(np.uint8)
    vector = grayscale_images_to_vectors(canonical_image[np.newaxis, ...])[0]
    return canonical_image, vector


def preprocess_character_image_array_legacy(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Giữ pipeline ảnh ngoài cũ để so sánh thực nghiệm trước/sau."""
    pixels = np.asarray(image)
    if (
        pixels.ndim != 2
        or pixels.size == 0
        or not np.issubdtype(pixels.dtype, np.number)
        or not np.isfinite(pixels).all()
        or np.any(pixels < 0)
        or np.any(pixels > 255)
    ):
        raise ValueError("Ảnh crop phải là grayscale không rỗng trong khoảng [0, 255].")
    pixels = pixels.astype(np.uint8)
    border = np.concatenate(
        (pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1])
    )

    if float(np.median(border)) > 127:
        pixels = 255 - pixels
        border = np.concatenate(
            (pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1])
        )

    background_level = float(np.median(border))
    pixels = np.clip(
        pixels.astype(np.float32) - background_level, 0, 255
    ).astype(np.uint8)
    peak_intensity = int(pixels.max())
    if peak_intensity == 0:
        raise ValueError("Ảnh không có nét ký tự rõ; hãy kiểm tra ảnh đầu vào.")

    threshold = max(8, int(peak_intensity * 0.12))
    foreground = pixels > threshold
    rows, columns = np.where(foreground)
    if rows.size == 0:
        raise ValueError("Ảnh không có nét ký tự rõ; hãy kiểm tra ảnh đầu vào.")

    cropped = Image.fromarray(pixels).crop(
        (int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1)
    )
    scale = min(20 / cropped.width, 20 / cropped.height)
    resized_size = (
        max(1, round(cropped.width * scale)),
        max(1, round(cropped.height * scale)),
    )
    cropped = cropped.resize(resized_size, Image.Resampling.LANCZOS)
    canvas = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT), color=0)
    offset = (
        (IMAGE_WIDTH - cropped.width) // 2,
        (IMAGE_HEIGHT - cropped.height) // 2,
    )
    canvas.paste(cropped, offset)
    canonical_image = np.asarray(canvas, dtype=np.uint8)
    vector = grayscale_images_to_vectors(canonical_image[np.newaxis, ...])[0]
    return canonical_image, vector
