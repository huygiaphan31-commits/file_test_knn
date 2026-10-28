"""Tạo biểu đồ từ kết quả đánh giá thực tế."""

from pathlib import Path

import numpy as np


def save_confusion_matrix(
    matrix: np.ndarray,
    class_ids: tuple[int, ...],
    character_mapping: dict[int, str],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(14, 12))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, label="Số mẫu")
    characters = [character_mapping[class_id] for class_id in class_ids]
    axis.set(
        title="Confusion matrix — EMNIST ByClass test subset",
        xlabel="Ký tự dự đoán",
        ylabel="Ký tự thật",
        xticks=np.arange(len(class_ids)),
        yticks=np.arange(len(class_ids)),
        xticklabels=characters,
        yticklabels=characters,
    )
    axis.tick_params(axis="x", labelsize=7, rotation=90)
    axis.tick_params(axis="y", labelsize=7)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_sample_predictions(
    images: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    selected_indices: np.ndarray,
    character_mapping: dict[int, str],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    columns = 4
    rows = (len(selected_indices) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows))
    axes_array = np.asarray(axes).reshape(-1)
    for axis, index in zip(axes_array, selected_indices):
        truth = int(true_labels[index])
        prediction = int(predicted_labels[index])
        correct = truth == prediction
        axis.imshow(images[index], cmap="gray", vmin=0, vmax=255)
        axis.set_title(
            f"Thật: {character_mapping[truth]} | Dự đoán: "
            f"{character_mapping[prediction]}\n{'Đúng' if correct else 'Sai'}",
            color="green" if correct else "red",
        )
        axis.axis("off")
    for axis in axes_array[len(selected_indices):]:
        axis.axis("off")

    figure.suptitle("Ví dụ từ tập test ByClass (subset)")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_query_preview(
    image: np.ndarray,
    predicted_character: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(3, 3))
    axis.imshow(image, cmap="gray", vmin=0, vmax=255)
    axis.set_title(f"Dự đoán: {predicted_character}")
    axis.axis("off")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
