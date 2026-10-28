"""Fixture IDX gzip nhỏ dùng riêng cho kiểm thử phần mềm."""

import gzip
import struct
from pathlib import Path

import numpy as np

ASCII_CODES = [*range(48, 58), *range(65, 91), *range(97, 123)]


def write_idx_pair(
    folder: Path,
    prefix: str,
    samples_by_class: dict[int, int],
) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    images_path = folder / f"{prefix}-images-idx3-ubyte.gz"
    labels_path = folder / f"{prefix}-labels-idx1-ubyte.gz"
    images: list[np.ndarray] = []
    labels: list[int] = []
    for class_id, count in samples_by_class.items():
        for sample_index in range(count):
            image = np.zeros((28, 28), dtype=np.uint8)
            if class_id == 0:
                image[2:8, 2:8] = 220
            else:
                image[20:26, 20:26] = 220
            image[(sample_index % 5) + 10, (sample_index % 5) + 10] = 100
            images.append(image)
            labels.append(class_id)
    with gzip.open(images_path, "wb") as image_file:
        image_file.write(struct.pack(">IIII", 2051, len(images), 28, 28))
        image_file.write(np.stack(images).tobytes())
    with gzip.open(labels_path, "wb") as label_file:
        label_file.write(struct.pack(">II", 2049, len(labels)))
        label_file.write(bytes(labels))
    return images_path, labels_path


def write_complete_mapping(path: Path) -> None:
    path.write_text(
        "".join(f"{class_id} {codepoint}\n" for class_id, codepoint in enumerate(ASCII_CODES)),
        encoding="ascii",
    )


def write_demo_fixture(
    folder: Path,
    train_per_class: int = 10,
    test_per_class: int = 5,
) -> None:
    write_idx_pair(
        folder,
        "emnist-byclass-train",
        {0: train_per_class, 1: train_per_class},
    )
    write_idx_pair(
        folder,
        "emnist-byclass-test",
        {0: test_per_class, 1: test_per_class},
    )
    write_complete_mapping(folder / "emnist-byclass-mapping.txt")
