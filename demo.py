"""Demo nhận diện chữ số viết tay 8x8 bằng KNN và scikit-learn."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Huấn luyện KNN trên load_digits và kiểm thử ảnh chữ số."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Số láng giềng K (mặc định: 3).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=8,
        help="Số ảnh kiểm thử riêng, từ 5 đến 10 (mặc định: 8).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Thư mục lưu biểu đồ và số liệu thực chạy (mặc định: results).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Mở các biểu đồ sau khi lưu (mặc định chỉ lưu file).",
    )
    args = parser.parse_args()

    if not 5 <= args.samples <= 10:
        parser.error("--samples phải nằm trong khoảng từ 5 đến 10.")
    if args.k < 1:
        parser.error("--k phải là số nguyên dương.")
    return args


def choose_test_examples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    count: int,
    random_state: int,
) -> np.ndarray:
    """Chọn mẫu tái lập được, ưu tiên có cả dự đoán đúng và sai."""
    rng = np.random.default_rng(random_state)
    correct = np.flatnonzero(y_true == y_pred)
    incorrect = np.flatnonzero(y_true != y_pred)

    selected: list[int] = []
    if correct.size:
        selected.append(int(rng.choice(correct)))
    if incorrect.size:
        selected.append(int(rng.choice(incorrect)))

    remaining = np.setdiff1d(np.arange(y_true.size), selected)
    extra_count = min(count - len(selected), remaining.size)
    if extra_count:
        selected.extend(
            int(index) for index in rng.choice(remaining, size=extra_count, replace=False)
        )

    result = np.asarray(selected, dtype=int)
    rng.shuffle(result)
    return result


def save_confusion_matrix(
    matrix: np.ndarray,
    output_path: Path,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Số ảnh")
    axis.set(
        title="Confusion matrix — tập kiểm thử",
        xlabel="Nhãn dự đoán",
        ylabel="Nhãn thật",
        xticks=np.arange(10),
        yticks=np.arange(10),
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(figure)


def save_sample_plot(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    selected: np.ndarray,
    output_path: Path,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    columns = 4
    rows = (selected.size + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows))
    axes_array = np.asarray(axes).reshape(-1)

    for axis, index in zip(axes_array, selected):
        truth = int(y_true[index])
        prediction = int(y_pred[index])
        is_correct = truth == prediction
        axis.imshow(images[index], cmap="gray", vmin=0, vmax=16)
        axis.set_title(
            f"Thật: {truth} | Dự đoán: {prediction}\n"
            f"{'Đúng' if is_correct else 'Sai'}",
            color="green" if is_correct else "red",
        )
        axis.axis("off")

    for axis in axes_array[selected.size :]:
        axis.axis("off")

    figure.suptitle("Mẫu kiểm thử lấy từ tập test của load_digits()")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(figure)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")

    try:
        digits = load_digits()
        images = digits.images
        labels = digits.target
        features = images.reshape(images.shape[0], -1)

        (
            x_train,
            x_test,
            y_train,
            y_test,
            _train_indices,
            test_indices,
            _train_images,
            test_images,
        ) = train_test_split(
            features,
            labels,
            np.arange(labels.size),
            images,
            test_size=0.25,
            random_state=42,
            stratify=labels,
        )

        if args.k > y_train.size:
            raise ValueError(
                f"--k ({args.k}) không được lớn hơn số mẫu train ({y_train.size})."
            )

        model = KNeighborsClassifier(n_neighbors=args.k)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        accuracy = float(accuracy_score(y_test, y_pred))
        matrix = confusion_matrix(y_test, y_pred, labels=np.arange(10))
        selected = choose_test_examples(y_test, y_pred, args.samples, random_state=42)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        matrix_path = args.output_dir / "confusion_matrix.png"
        samples_path = args.output_dir / "test_samples.png"
        metrics_path = args.output_dir / "metrics.json"

        save_confusion_matrix(matrix, matrix_path, args.show)
        save_sample_plot(
            test_images,
            y_test,
            y_pred,
            selected,
            samples_path,
            args.show,
        )

        sample_results: list[dict[str, Any]] = []
        for index in selected:
            sample_results.append(
                {
                    "dataset_index": int(test_indices[index]),
                    "true_label": int(y_test[index]),
                    "predicted_label": int(y_pred[index]),
                    "status": "correct" if y_test[index] == y_pred[index] else "incorrect",
                }
            )

        results = {
            "dataset": "sklearn.datasets.load_digits",
            "image_shape_pixels": list(images.shape[1:]),
            "features_per_image": int(features.shape[1]),
            "split": {
                "test_size": 0.25,
                "random_state": 42,
                "stratified": True,
                "train_samples": int(y_train.size),
                "test_samples": int(y_test.size),
            },
            "model": {
                "name": "KNeighborsClassifier",
                "n_neighbors": args.k,
                "distance": "Euclidean (scikit-learn default Minkowski p=2)",
            },
            "accuracy": accuracy,
            "confusion_matrix_labels": list(range(10)),
            "confusion_matrix": matrix.tolist(),
            "sample_tests": sample_results,
            "note": (
                "Các ảnh mẫu được lấy từ tập test có nhãn của load_digits(); "
                "đây không phải ảnh tự chụp."
            ),
        }
        metrics_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print("Dataset: sklearn.datasets.load_digits()")
        print(f"Kích thước mỗi ảnh: {images.shape[1]}x{images.shape[2]} pixel (8x8)")
        print(f"Đặc trưng mỗi ảnh sau khi làm phẳng: {features.shape[1]}")
        print(f"Số mẫu: train={y_train.size}, test={y_test.size}")
        print(f"KNN: k={args.k}; accuracy trên tập test={accuracy:.4f} ({accuracy:.2%})")
        print("Confusion matrix (hàng=nhãn thật, cột=nhãn dự đoán):")
        print(matrix)
        print(f"\nKiểm thử riêng {selected.size} ảnh thuộc tập test:")
        for result in sample_results:
            status = "ĐÚNG" if result["status"] == "correct" else "SAI"
            print(
                f"- index={result['dataset_index']}: "
                f"thật={result['true_label']}, "
                f"dự đoán={result['predicted_label']} — {status}"
            )
        print("\nĐây là kết quả chạy thực tế trên dữ liệu của dataset, không phải ảnh tự chụp.")
        print(f"Đã lưu: {matrix_path}, {samples_path}, {metrics_path}")
    except (OSError, ValueError) as error:
        raise SystemExit(f"Lỗi khi chạy demo: {error}") from error


if __name__ == "__main__":
    main()
