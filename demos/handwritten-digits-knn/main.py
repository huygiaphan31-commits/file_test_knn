"""Chạy đánh giá EMNIST ByClass và tùy chọn dự đoán một ảnh ký tự."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

from config import (
    DATA_DIR,
    DEFAULT_CLASS_IDS,
    K_CANDIDATES,
    MAX_TEST_PER_CLASS,
    MAX_TRAIN_PER_CLASS,
    MAX_VALIDATION_PER_CLASS,
    RANDOM_STATE,
    RESULTS_DIR,
    SAMPLE_IMAGE_COUNT,
    VALIDATION_FRACTION,
)
from data_loader import IDXDataset, load_emnist_byclass, load_emnist_training
from evaluation import (
    evaluate,
    save_confusion_matrix_csv,
    save_predictions_csv,
    select_k,
    split_train_validation,
)
from knn import ManualKNN
from line_recognition import recognize_line, save_segmentation_debug
from preprocessing import (
    dataset_images_to_vectors,
    load_grayscale_image,
    orient_emnist_images,
)
from visualization import (
    save_confusion_matrix,
    save_query_preview,
    save_sample_predictions,
)
import numpy as np


def parse_class_ids(value: str) -> tuple[int, ...]:
    try:
        class_ids = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Class ID phải là số nguyên, cách nhau bằng dấu phẩy."
        ) from error
    if not class_ids or len(set(class_ids)) != len(class_ids):
        raise argparse.ArgumentTypeError("Cần các class ID duy nhất, không rỗng.")
    if any(class_id < 0 or class_id >= 62 for class_id in class_ids):
        raise argparse.ArgumentTypeError("Class ID ByClass phải từ 0 đến 61.")
    return class_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tự cài k-NN nhận diện một ký tự hoặc một dòng chữ viết tay "
            "bằng EMNIST ByClass."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Thư mục chứa các file IDX gzip và mapping của ByClass.",
    )
    parser.add_argument(
        "--classes",
        type=parse_class_ids,
        default=DEFAULT_CLASS_IDS,
        help="Class ID cần dùng; mặc định đủ 62 lớp ByClass.",
    )
    parser.add_argument(
        "--max-train-per-class",
        type=int,
        default=MAX_TRAIN_PER_CLASS,
        help=f"Giới hạn reservoir train cho mỗi lớp (mặc định {MAX_TRAIN_PER_CLASS}).",
    )
    parser.add_argument(
        "--max-test-per-class",
        type=int,
        default=MAX_TEST_PER_CLASS,
        help=f"Giới hạn official test cho mỗi lớp (mặc định {MAX_TEST_PER_CLASS}).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=SAMPLE_IMAGE_COUNT,
        help="Số ví dụ test lưu thành hình (5–10).",
    )
    parser.add_argument(
        "--predict-image",
        type=Path,
        help="Đường dẫn một ảnh ký tự hoặc một dòng ký tự.",
    )
    parser.add_argument(
        "--predict-images",
        type=Path,
        nargs="+",
        help="Một hoặc nhiều ảnh ký tự/dòng ký tự để dự đoán bằng cùng model.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Chỉ nạp train, chọn k và dự đoán; không đọc/đánh giá test subset.",
    )
    parser.add_argument(
        "--debug-segmentation",
        action="store_true",
        help="Lưu mặt nạ, vùng có đánh số và crop ký tự vào thư mục debug riêng.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Thư mục lưu kết quả lần chạy này.",
    )
    return parser


def fit_classifier(
    training_data: IDXDataset,
    class_ids: tuple[int, ...],
    mapping: dict[int, str],
    k_candidates: tuple[int, ...] = K_CANDIDATES,
) -> tuple[
    ManualKNN,
    int,
    dict[int, float],
    np.ndarray,
    np.ndarray,
    tuple[int, ...],
]:
    label_order = tuple(sorted(class_ids, key=lambda class_id: (mapping[class_id], class_id)))
    training_vectors = dataset_images_to_vectors(training_data.images)
    training_indices, validation_indices = split_train_validation(
        training_data.labels,
        class_ids,
        VALIDATION_FRACTION,
        MAX_VALIDATION_PER_CLASS,
        RANDOM_STATE,
    )
    candidate_k = tuple(k for k in k_candidates if k <= len(training_indices))
    if not candidate_k:
        if not k_candidates:
            raise ValueError("Cần ít nhất một giá trị k để đánh giá.")
        raise ValueError(
            f"Không đủ mẫu train sau khi giữ validation để dùng k={max(k_candidates)}; "
            f"còn {len(training_indices)} mẫu. k không được tự động giảm."
        )

    selected_k, validation_scores = select_k(
        training_vectors[training_indices],
        training_data.labels[training_indices],
        training_vectors[validation_indices],
        training_data.labels[validation_indices],
        class_ids,
        label_order,
        candidate_k,
    )
    model = ManualKNN(training_vectors, training_data.labels, label_order)
    return (
        model,
        selected_k,
        validation_scores,
        training_indices,
        validation_indices,
        candidate_k,
    )


def predict_external_images(
    image_paths: list[Path],
    model: ManualKNN,
    selected_k: int,
    mapping: dict[int, str],
    results_dir: Path,
    debug_segmentation: bool = False,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    multiple_images = len(image_paths) > 1
    for index, image_path in enumerate(image_paths, start=1):
        try:
            result = recognize_line(image_path, model, selected_k, mapping)
            if debug_segmentation:
                debug_root = results_dir / "segmentation_debug" / image_path.stem
                artifacts = save_segmentation_debug(image_path, debug_root, result)
                print(f"Segmentation debug saved: {artifacts[0].parent}")
        except (OSError, ValueError) as error:
            print(f"Lỗi ảnh '{image_path}': {error}")
            continue
        if not result.text:
            print(f"Ảnh '{image_path}': Không tìm thấy ký tự.")
            continue

        preview_name = (
            f"query_prediction_{index:02d}.png"
            if multiple_images
            else "query_prediction.png"
        )
        save_query_preview(
            np.asarray(load_grayscale_image(image_path)),
            result.text,
            results_dir / preview_name,
        )
        print(f"Ảnh '{image_path}'; predicted character: {result.text!r}")


def select_example_indices(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    count: int,
    random_state: int,
) -> np.ndarray:
    if not 5 <= count <= 10:
        raise ValueError("Số ví dụ phải nằm trong khoảng 5–10.")
    if len(true_labels) < count:
        raise ValueError(f"Cần tối thiểu {count} ảnh test, hiện có {len(true_labels)}.")

    rng = np.random.default_rng(random_state)
    correct = np.flatnonzero(true_labels == predicted_labels)
    incorrect = np.flatnonzero(true_labels != predicted_labels)
    selected: list[int] = []
    if correct.size:
        selected.append(int(rng.choice(correct)))
    if incorrect.size:
        selected.append(int(rng.choice(incorrect)))
    remaining = np.setdiff1d(np.arange(len(true_labels)), selected)
    selected.extend(
        int(index)
        for index in rng.choice(remaining, size=count - len(selected), replace=False)
    )
    result = np.asarray(selected, dtype=np.int64)
    rng.shuffle(result)
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args()
    image_paths = ([args.predict_image] if args.predict_image else []) + (
        args.predict_images or []
    )
    if args.skip_evaluation and not image_paths:
        raise SystemExit("--skip-evaluation cần ít nhất một ảnh dự đoán.")
    if args.max_train_per_class < 2:
        raise SystemExit("--max-train-per-class phải tối thiểu là 2.")
    if args.max_test_per_class < 1:
        raise SystemExit("--max-test-per-class phải là số dương.")
    if not 5 <= args.samples <= 10:
        raise SystemExit("--samples phải từ 5 đến 10.")

    matplotlib.use("Agg")
    try:
        if args.skip_evaluation:
            mapping, training_data = load_emnist_training(
                args.data_dir,
                args.classes,
                args.max_train_per_class,
                RANDOM_STATE,
            )
            (
                model,
                selected_k,
                _,
                _,
                _,
                _,
            ) = fit_classifier(training_data, args.classes, mapping)
            print(
                f"Prediction-only mode: selected k={selected_k}; "
                f"training subset={len(training_data.labels)}"
            )
            predict_external_images(
                image_paths,
                model,
                selected_k,
                mapping,
                args.results_dir,
                args.debug_segmentation,
            )
            return

        mapping, training_data, test_data = load_emnist_byclass(
            args.data_dir,
            args.classes,
            args.max_train_per_class,
            args.max_test_per_class,
            RANDOM_STATE,
        )
        (
            model,
            selected_k,
            validation_scores,
            training_indices,
            validation_indices,
            candidate_k,
        ) = fit_classifier(training_data, args.classes, mapping)
        test_vectors = dataset_images_to_vectors(test_data.images)
        predicted_test, _ = model.predict_many(test_vectors, selected_k)
        metrics = evaluate(test_data.labels, predicted_test, args.classes)
        selected_examples = select_example_indices(
            test_data.labels,
            predicted_test,
            args.samples,
            RANDOM_STATE,
        )

        args.results_dir.mkdir(parents=True, exist_ok=True)
        save_confusion_matrix(
            np.asarray(metrics["confusion_matrix"]),
            args.classes,
            mapping,
            args.results_dir / "confusion_matrix.png",
        )
        save_confusion_matrix_csv(
            args.results_dir / "confusion_matrix.csv",
            np.asarray(metrics["confusion_matrix"]),
            args.classes,
            mapping,
        )
        save_sample_predictions(
            orient_emnist_images(test_data.images),
            test_data.labels,
            predicted_test,
            selected_examples,
            mapping,
            args.results_dir / "test_examples.png",
        )
        save_predictions_csv(
            args.results_dir / "test_predictions.csv",
            test_data.sample_ids,
            test_data.labels,
            predicted_test,
            mapping,
        )

        result_payload = {
            "dataset": "EMNIST ByClass",
            "source_note": (
                "Official train/test IDX gzip files; metrics use a deterministic "
                "per-class capped subset, not necessarily the full benchmark."
            ),
            "class_ids": list(args.classes),
            "class_mapping": {
                str(class_id): mapping[class_id] for class_id in args.classes
            },
            "image_shape": [28, 28],
            "feature_count": 28 * 28,
            "preprocessing": (
                "EMNIST orientation transpose, grayscale pixel scale [0,1], "
                "row-major flatten; user image is cropped/centered on 28x28."
            ),
            "split": {
                "random_state": RANDOM_STATE,
                "validation_fraction": VALIDATION_FRACTION,
                "max_validation_per_class": MAX_VALIDATION_PER_CLASS,
                "train_sampled_count": int(len(training_data.labels)),
                "train_used_for_k_selection": int(len(training_indices)),
                "validation_count": int(len(validation_indices)),
                "official_test_sampled_count": int(len(test_data.labels)),
                "train_observed_class_counts": training_data.observed_class_counts,
                "test_observed_class_counts": test_data.observed_class_counts,
            },
            "algorithm": {
                "name": "Manual Euclidean k-NN",
                "distance_formula": "sqrt(sum((x_l-y_l)^2))",
                "tie_break": (
                    "minimum sum of neighbor distances among tied labels, then "
                    "fixed character order from mapping"
                ),
                "k_candidates": list(candidate_k),
                "validation_macro_f1_by_k": {
                    str(k): score for k, score in validation_scores.items()
                },
                "selected_k": selected_k,
            },
            "test_metrics": metrics,
            "sample_predictions": [
                {
                    "dataset_sample_id": int(test_data.sample_ids[index]),
                    "true_class_id": int(test_data.labels[index]),
                    "true_character": mapping[int(test_data.labels[index])],
                    "predicted_class_id": int(predicted_test[index]),
                    "predicted_character": mapping[int(predicted_test[index])],
                    "status": (
                        "correct"
                        if test_data.labels[index] == predicted_test[index]
                        else "incorrect"
                    ),
                }
                for index in selected_examples
            ],
            "limitation": (
                "EMNIST labels digits and Latin uppercase/lowercase letters; "
                "Vietnamese diacritics are not included. Results on capped "
                "ByClass samples do not establish performance on real handwriting."
            ),
        }
        (args.results_dir / "metrics.json").write_text(
            json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print("Dataset: EMNIST ByClass")
        print(f"Class count: {len(args.classes)}")
        print(
            f"Samples: train subset={len(training_data.labels)}, "
            f"validation={len(validation_indices)}, test subset={len(test_data.labels)}"
        )
        print(f"Selected k={selected_k}; test accuracy={metrics['accuracy']:.4f}")
        print(f"Test macro-F1={metrics['macro_f1']:.4f}")
        print("Evaluation is on the capped official test subset, not a full-set score.")
        print(f"Actual run artifacts saved to {args.results_dir}")

        if image_paths:
            predict_external_images(
                image_paths,
                model,
                selected_k,
                mapping,
                args.results_dir,
                args.debug_segmentation,
            )
    except (FileNotFoundError, OSError, ValueError) as error:
        raise SystemExit(f"Lỗi demo: {error}") from error


if __name__ == "__main__":
    main()
