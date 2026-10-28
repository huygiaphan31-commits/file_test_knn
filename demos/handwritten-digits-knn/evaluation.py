"""Chia validation, chọn K và tính chỉ số không dùng estimator k-NN."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from config import MAX_VALIDATION_PER_CLASS
from knn import ManualKNN


def split_train_validation(
    labels: np.ndarray,
    class_ids: tuple[int, ...],
    validation_fraction: float,
    max_validation_per_class: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Tạo chỉ số train/validation phân tầng; giới hạn validation mỗi lớp."""
    labels = np.asarray(labels, dtype=np.int64)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction phải nằm giữa 0 và 1.")
    if max_validation_per_class < 1:
        raise ValueError("max_validation_per_class phải là số dương.")

    rng = np.random.default_rng(random_state)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for class_id in class_ids:
        class_indices = np.flatnonzero(labels == class_id)
        if class_indices.size < 2:
            raise ValueError(
                f"Cần ít nhất 2 mẫu train đã lấy cho class ID {class_id} "
                "để tạo validation."
            )
        rng.shuffle(class_indices)
        validation_count = max(1, int(round(class_indices.size * validation_fraction)))
        validation_count = min(validation_count, class_indices.size - 1)
        class_validation = class_indices[:validation_count]
        if class_validation.size > max_validation_per_class:
            class_validation = rng.choice(
                class_validation, size=max_validation_per_class, replace=False
            )
        selected_validation = set(int(index) for index in class_validation)
        validation_indices.extend(sorted(selected_validation))
        train_indices.extend(
            int(index)
            for index in class_indices
            if int(index) not in selected_validation
        )
    return (
        np.asarray(train_indices, dtype=np.int64),
        np.asarray(validation_indices, dtype=np.int64),
    )


def select_k(
    training_vectors: np.ndarray,
    training_labels: np.ndarray,
    validation_vectors: np.ndarray,
    validation_labels: np.ndarray,
    class_ids: tuple[int, ...],
    label_order: tuple[int, ...],
    candidates: tuple[int, ...],
) -> tuple[int, dict[int, float]]:
    """Chọn K có macro-F1 validation cao nhất; hòa điểm thì K nhỏ hơn."""
    if not candidates or any(k < 1 or k > len(training_labels) for k in candidates):
        raise ValueError("Các giá trị K phải dương và không lớn hơn số mẫu train.")
    if validation_vectors.shape[0] == 0:
        raise ValueError("Validation không được rỗng.")

    model = ManualKNN(training_vectors, training_labels, label_order)
    max_k = max(candidates)
    predictions_by_k = {k: [] for k in candidates}
    for vector in validation_vectors:
        nearest = model.find_neighbors(vector, max_k)
        for k in candidates:
            prediction = model.classify_neighbors(
                nearest.labels[:k], nearest.distances[:k]
            )
            predictions_by_k[k].append(prediction)

    scores = {
        k: macro_f1(
            validation_labels,
            np.asarray(predictions_by_k[k], dtype=np.int64),
            class_ids,
        )
        for k in candidates
    }
    best_k = min(scores, key=lambda k: (-scores[k], k))
    return best_k, scores


def confusion_matrix(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_ids: tuple[int, ...],
) -> np.ndarray:
    """Tạo ma trận nhầm lẫn theo thứ tự class_ids."""
    y_true = np.asarray(true_labels, dtype=np.int64)
    y_predicted = np.asarray(predicted_labels, dtype=np.int64)
    if y_true.ndim != 1 or y_predicted.shape != y_true.shape or y_true.size == 0:
        raise ValueError("Nhãn thật/dự đoán phải là vector cùng kích thước, không rỗng.")
    positions = {class_id: index for index, class_id in enumerate(class_ids)}
    matrix = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
    for truth, prediction in zip(y_true, y_predicted):
        if int(truth) not in positions or int(prediction) not in positions:
            raise ValueError("Nhãn thật hoặc dự đoán nằm ngoài class_ids.")
        matrix[positions[int(truth)], positions[int(prediction)]] += 1
    return matrix


def macro_f1(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_ids: tuple[int, ...],
) -> float:
    """Tính trung bình F1 theo lớp; lớp không có dự đoán/nhãn được tính là 0."""
    matrix = confusion_matrix(true_labels, predicted_labels, class_ids)
    f1_scores: list[float] = []
    for index in range(len(class_ids)):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(
            0.0 if denominator == 0 else (2 * true_positive) / denominator
        )
    return float(np.mean(f1_scores))


def evaluate(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_ids: tuple[int, ...],
) -> dict[str, object]:
    matrix = confusion_matrix(true_labels, predicted_labels, class_ids)
    return {
        "accuracy": float(np.mean(np.asarray(true_labels) == np.asarray(predicted_labels))),
        "macro_f1": macro_f1(true_labels, predicted_labels, class_ids),
        "confusion_matrix": matrix.tolist(),
        "class_ids": list(class_ids),
    }


def save_predictions_csv(
    output_path: Path,
    sample_ids: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    character_mapping: dict[int, str],
) -> None:
    """Lưu nhãn thật/dự đoán cho từng mẫu test đã đánh giá."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["dataset_sample_id", "true_class_id", "true_character",
             "predicted_class_id", "predicted_character", "status"]
        )
        for sample_id, truth, prediction in zip(
            sample_ids, true_labels, predicted_labels
        ):
            writer.writerow(
                [
                    int(sample_id),
                    int(truth),
                    character_mapping[int(truth)],
                    int(prediction),
                    character_mapping[int(prediction)],
                    "correct" if truth == prediction else "incorrect",
                ]
            )


def save_confusion_matrix_csv(
    output_path: Path,
    matrix: np.ndarray,
    class_ids: tuple[int, ...],
    character_mapping: dict[int, str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\predicted"] + [character_mapping[i] for i in class_ids])
        for class_id, row in zip(class_ids, matrix):
            writer.writerow([character_mapping[class_id], *map(int, row)])
