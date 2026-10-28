"""Tự cài đặt k-NN: khoảng cách Euclid, sắp xếp và bỏ phiếu."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NeighborResult:
    """Các láng giềng gần nhất của một vector query."""

    indices: np.ndarray
    labels: np.ndarray
    distances: np.ndarray


def euclidean_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Tính d(x,y)=sqrt(sum((x_l-y_l)^2)) theo công thức giáo trình."""
    first_vector = np.asarray(first, dtype=np.float64).reshape(-1)
    second_vector = np.asarray(second, dtype=np.float64).reshape(-1)
    if first_vector.shape != second_vector.shape:
        raise ValueError("Hai vector phải có cùng số thuộc tính.")
    if first_vector.size == 0:
        raise ValueError("Vector đặc trưng không được rỗng.")
    difference = first_vector - second_vector
    return float(np.sqrt(np.sum(difference * difference)))


class ManualKNN:
    """Phân loại bằng k láng giềng, không dùng thư viện estimator k-NN."""

    def __init__(
        self,
        training_vectors: np.ndarray,
        training_labels: np.ndarray,
        label_order: tuple[int, ...],
    ) -> None:
        vectors = np.asarray(training_vectors, dtype=np.float32)
        labels = np.asarray(training_labels, dtype=np.int64)
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError("Dữ liệu train phải là ma trận 2D không rỗng.")
        if labels.ndim != 1 or labels.size != vectors.shape[0]:
            raise ValueError("Số nhãn phải khớp số vector train.")
        if not np.isfinite(vectors).all():
            raise ValueError("Vector train chứa giá trị NaN hoặc vô hạn.")
        if not label_order or len(set(label_order)) != len(label_order):
            raise ValueError("label_order phải chứa các mã nhãn duy nhất.")

        self.training_vectors = np.ascontiguousarray(vectors)
        self.training_labels = labels
        self.label_order = tuple(label_order)
        self._label_rank = {
            label: index for index, label in enumerate(self.label_order)
        }

    def find_neighbors(self, query: np.ndarray, k: int) -> NeighborResult:
        """Tính Euclid tới mọi mẫu train, sắp xếp tăng dần và lấy k mẫu đầu."""
        if not 1 <= k <= self.training_labels.size:
            raise ValueError(
                f"k phải nằm trong khoảng 1–{self.training_labels.size}."
            )
        query_vector = np.asarray(query, dtype=np.float32).reshape(-1)
        if query_vector.size != self.training_vectors.shape[1]:
            raise ValueError(
                "Số thuộc tính query phải khớp số thuộc tính của vector train."
            )
        if not np.isfinite(query_vector).all():
            raise ValueError("Vector query chứa giá trị NaN hoặc vô hạn.")

        # Mỗi hàng là hiệu giữa query và một mẫu; phép tổng bình phương rồi căn
        # bậc hai thực hiện đúng công thức Euclid theo từng thuộc tính.
        differences = self.training_vectors - query_vector
        np.square(differences, out=differences)
        distances = np.sqrt(np.sum(differences, axis=1, dtype=np.float64))
        nearest_indices = np.argsort(distances, kind="stable")[:k]
        return NeighborResult(
            indices=nearest_indices,
            labels=self.training_labels[nearest_indices],
            distances=distances[nearest_indices],
        )

    def classify_neighbors(
        self,
        neighbor_labels: np.ndarray,
        neighbor_distances: np.ndarray,
    ) -> int:
        """Bỏ phiếu đa số, phá hòa theo tổng khoảng cách rồi thứ tự nhãn."""
        labels = np.asarray(neighbor_labels, dtype=np.int64)
        distances = np.asarray(neighbor_distances, dtype=np.float64)
        if labels.ndim != 1 or labels.size == 0 or distances.shape != labels.shape:
            raise ValueError("Danh sách láng giềng và khoảng cách không hợp lệ.")

        vote_counts = Counter(int(label) for label in labels)
        highest_votes = max(vote_counts.values())
        tied_labels = [
            label for label, count in vote_counts.items() if count == highest_votes
        ]
        if len(tied_labels) == 1:
            return tied_labels[0]

        distance_sums = {
            label: float(distances[labels == label].sum()) for label in tied_labels
        }
        minimum_sum = min(distance_sums.values())
        closest_labels = [
            label
            for label in tied_labels
            if np.isclose(distance_sums[label], minimum_sum, rtol=1e-12, atol=1e-12)
        ]
        return min(
            closest_labels,
            key=lambda label: (self._label_rank.get(label, len(self._label_rank)), label),
        )

    def predict_one(self, query: np.ndarray, k: int) -> tuple[int, NeighborResult]:
        neighbors = self.find_neighbors(query, k)
        prediction = self.classify_neighbors(neighbors.labels, neighbors.distances)
        return prediction, neighbors

    def predict_many(
        self,
        queries: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, list[NeighborResult]]:
        """Dự đoán lần lượt các query để giới hạn bộ nhớ trung gian."""
        query_matrix = np.asarray(queries, dtype=np.float32)
        if query_matrix.ndim != 2:
            raise ValueError("Queries phải là ma trận 2D.")
        predictions: list[int] = []
        all_neighbors: list[NeighborResult] = []
        for query in query_matrix:
            prediction, neighbors = self.predict_one(query, k)
            predictions.append(prediction)
            all_neighbors.append(neighbors)
        return np.asarray(predictions, dtype=np.int64), all_neighbors
