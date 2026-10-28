"""Đọc các file IDX gzip và mapping ký tự của EMNIST ByClass."""

from __future__ import annotations

import gzip
import random
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import (
    DEFAULT_CLASS_IDS,
    IDX_FILENAMES,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAX_TEST_PER_CLASS,
    MAX_TRAIN_PER_CLASS,
    NUM_CLASSES,
)


@dataclass(frozen=True)
class IDXDataset:
    """Một phần dữ liệu IDX đã lấy mẫu, kèm chỉ số gốc và thống kê lớp."""

    images: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    observed_class_counts: dict[int, int]


def resolve_dataset_files(
    dataset_dir: Path,
    required_keys: tuple[str, ...] = tuple(IDX_FILENAMES),
) -> dict[str, Path]:
    """Tìm các file EMNIST trong thư mục gốc hoặc thư mục con cùng tên."""
    root = dataset_dir.expanduser().resolve()
    candidates = (root, root / "gzip", root / "emnist-byclass")
    files: dict[str, Path] = {}
    missing: list[str] = []
    for key in required_keys:
        filename = IDX_FILENAMES[key]
        match = next(
            (directory / filename for directory in candidates if (directory / filename).is_file()),
            None,
        )
        if match is None:
            missing.append(filename)
        else:
            files[key] = match
    if missing:
        raise FileNotFoundError(
            "Thiếu file EMNIST ByClass: "
            + ", ".join(missing)
            + f". Hãy kiểm tra archive đã giải nén trong {root}; xem data/README.md."
        )
    return files


def load_character_mapping(mapping_path: Path) -> dict[int, str]:
    """Đọc và xác thực ánh xạ class ID -> 62 ký tự EMNIST ByClass."""
    mapping: dict[int, str] = {}
    with mapping_path.open("r", encoding="ascii") as file:
        for line_number, line in enumerate(file, start=1):
            content = line.partition("#")[0].strip()
            if not content:
                continue
            parts = content.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Dòng {line_number} trong mapping phải có class_id và ASCII code."
                )
            try:
                class_id, codepoint = map(int, parts)
            except ValueError as error:
                raise ValueError(f"Dòng mapping không hợp lệ: {line_number}") from error
            if not 0 <= class_id < NUM_CLASSES:
                raise ValueError(f"class_id ngoài khoảng 0–{NUM_CLASSES - 1}: {class_id}")
            if not 0 <= codepoint <= 0x10FFFF:
                raise ValueError(f"Mã Unicode không hợp lệ ở dòng {line_number}.")
            if class_id in mapping:
                raise ValueError(f"class_id {class_id} bị lặp trong mapping.")
            mapping[class_id] = chr(codepoint)

    if set(mapping) != set(range(NUM_CLASSES)):
        raise ValueError(
            f"Mapping phải có đúng {NUM_CLASSES} class ID; "
            f"đã đọc {len(mapping)} lớp."
        )
    expected_characters = (
        set("0123456789")
        | set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        | set("abcdefghijklmnopqrstuvwxyz")
    )
    actual_characters = list(mapping.values())
    if len(set(actual_characters)) != NUM_CLASSES:
        raise ValueError("Mapping có ký tự bị trùng.")
    if set(actual_characters) != expected_characters:
        missing_characters = sorted(expected_characters - set(actual_characters))
        unexpected_characters = sorted(set(actual_characters) - expected_characters)
        raise ValueError(
            "Mapping phải chứa đúng các ký tự 0-9, A-Z và a-z; "
            f"thiếu={missing_characters}, ngoài tập={unexpected_characters}."
        )
    return mapping


def _read_exact(stream, size: int, description: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(f"File IDX bị cắt cụt khi đọc {description}.")
    return data


def load_idx_subset(
    images_path: Path,
    labels_path: Path,
    class_ids: tuple[int, ...] = DEFAULT_CLASS_IDS,
    max_per_class: int = MAX_TRAIN_PER_CLASS,
    random_state: int = 42,
) -> IDXDataset:
    """Đọc IDX theo luồng và lấy mẫu reservoir có seed cho từng lớp."""
    if not class_ids or len(set(class_ids)) != len(class_ids):
        raise ValueError("class_ids phải chứa ít nhất một mã lớp duy nhất.")
    if any(class_id < 0 or class_id >= NUM_CLASSES for class_id in class_ids):
        raise ValueError(f"Mã lớp phải nằm trong khoảng 0–{NUM_CLASSES - 1}.")
    if max_per_class < 1:
        raise ValueError("max_per_class phải là số dương.")

    selected_classes = set(class_ids)
    class_images: dict[int, list[tuple[int, bytes]]] = {
        class_id: [] for class_id in class_ids
    }
    observed = {class_id: 0 for class_id in class_ids}
    rng = random.Random(random_state)
    image_size = IMAGE_HEIGHT * IMAGE_WIDTH

    try:
        with gzip.open(images_path, "rb") as image_file, gzip.open(
            labels_path, "rb"
        ) as label_file:
            image_magic, image_count, rows, columns = struct.unpack(
                ">IIII", _read_exact(image_file, 16, "image header")
            )
            label_magic, label_count = struct.unpack(
                ">II", _read_exact(label_file, 8, "label header")
            )
            if image_magic != 2051 or label_magic != 2049:
                raise ValueError("IDX magic number không đúng định dạng image/label.")
            if image_count != label_count:
                raise ValueError(
                    f"Số ảnh ({image_count}) khác số nhãn ({label_count})."
                )
            if (rows, columns) != (IMAGE_HEIGHT, IMAGE_WIDTH):
                raise ValueError(
                    f"EMNIST ByClass cần ảnh {IMAGE_HEIGHT}x{IMAGE_WIDTH}; "
                    f"file có kích thước {rows}x{columns}."
                )

            for sample_id in range(image_count):
                pixels = _read_exact(image_file, image_size, f"ảnh {sample_id}")
                label = _read_exact(label_file, 1, f"nhãn {sample_id}")[0]
                if label >= NUM_CLASSES:
                    raise ValueError(
                        f"Nhãn {label} ở mẫu {sample_id} ngoài ByClass."
                    )
                if label not in selected_classes:
                    continue

                observed[label] += 1
                bucket = class_images[label]
                entry = (sample_id, pixels)
                if len(bucket) < max_per_class:
                    bucket.append(entry)
                else:
                    replacement_index = rng.randrange(observed[label])
                    if replacement_index < max_per_class:
                        bucket[replacement_index] = entry
    except (OSError, EOFError, zlib.error) as error:
        raise ValueError(f"Không thể đọc file IDX gzip: {error}") from error

    missing = [class_id for class_id in class_ids if not class_images[class_id]]
    if missing:
        raise ValueError(f"Không có mẫu cho class ID {missing} trong IDX.")

    sample_ids: list[int] = []
    labels: list[int] = []
    images: list[np.ndarray] = []
    for class_id in class_ids:
        for sample_id, pixels in class_images[class_id]:
            sample_ids.append(sample_id)
            labels.append(class_id)
            images.append(
                np.frombuffer(pixels, dtype=np.uint8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
            )

    return IDXDataset(
        images=np.stack(images),
        labels=np.asarray(labels, dtype=np.int64),
        sample_ids=np.asarray(sample_ids, dtype=np.int64),
        observed_class_counts=observed,
    )


def load_emnist_byclass(
    dataset_dir: Path,
    class_ids: tuple[int, ...] = DEFAULT_CLASS_IDS,
    max_train_per_class: int = MAX_TRAIN_PER_CLASS,
    max_test_per_class: int = MAX_TEST_PER_CLASS,
    random_state: int = 42,
) -> tuple[dict[int, str], IDXDataset, IDXDataset]:
    """Đọc mapping và lấy subset riêng từ official train/test IDX."""
    mapping, train_data = load_emnist_training(
        dataset_dir, class_ids, max_train_per_class, random_state
    )
    files = resolve_dataset_files(dataset_dir)
    test_data = load_idx_subset(
        files["test_images"],
        files["test_labels"],
        class_ids,
        max_test_per_class,
        random_state + 1,
    )
    return mapping, train_data, test_data


def load_emnist_training(
    dataset_dir: Path,
    class_ids: tuple[int, ...] = DEFAULT_CLASS_IDS,
    max_train_per_class: int = MAX_TRAIN_PER_CLASS,
    random_state: int = 42,
) -> tuple[dict[int, str], IDXDataset]:
    """Đọc mapping và subset train mà không mở file test."""
    files = resolve_dataset_files(
        dataset_dir, ("train_images", "train_labels", "mapping")
    )
    mapping = load_character_mapping(files["mapping"])
    absent_mapping = set(class_ids) - set(mapping)
    if absent_mapping:
        raise ValueError(f"Thiếu class ID trong mapping: {sorted(absent_mapping)}")
    train_data = load_idx_subset(
        files["train_images"],
        files["train_labels"],
        class_ids,
        max_train_per_class,
        random_state,
    )
    return mapping, train_data
