"""Run a paired legacy-versus-shared preprocessing experiment on train only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from config import (
    DATA_DIR,
    DEFAULT_CLASS_IDS,
    K_CANDIDATES,
    MAX_TRAIN_PER_CLASS,
    MAX_VALIDATION_PER_CLASS,
    RANDOM_STATE,
    RESULTS_DIR,
    VALIDATION_FRACTION,
)
from data_loader import load_emnist_training
from evaluation import select_k, split_train_validation
from knn import ManualKNN
from preprocessing import (
    grayscale_images_to_vectors,
    normalize_glyph,
    orient_emnist_images,
    preprocess_character_image_array_legacy,
)


def _reserve_output_dir(path: Path) -> Path:
    candidate = path
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}_{suffix:02d}")
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _diagnostics(image: np.ndarray, vector: np.ndarray) -> dict[str, object]:
    foreground = image > max(8, int(image.max() * 0.12))
    rows, columns = np.where(foreground)
    if rows.size:
        box = [
            int(columns.min()),
            int(rows.min()),
            int(columns.max()) + 1,
            int(rows.max()) + 1,
        ]
        center = [
            (box[0] + box[2] - 1) / 2,
            (box[1] + box[3] - 1) / 2,
        ]
    else:
        box = None
        center = None
    return {
        "shape": list(image.shape),
        "foreground_bbox_xyxy": box,
        "foreground_bbox_center_xy": center,
        "foreground_pixels": int(rows.size),
        "vector_length": int(vector.size),
        "vector_min": float(vector.min()),
        "vector_max": float(vector.max()),
    }


def _save_pair(
    folder: Path,
    name: str,
    legacy_image: np.ndarray,
    shared_image: np.ndarray,
) -> None:
    Image.fromarray(legacy_image, mode="L").save(folder / f"{name}_legacy.png")
    Image.fromarray(shared_image, mode="L").save(folder / f"{name}_shared.png")


def _vectors_and_images(
    raw_images: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    oriented = orient_emnist_images(raw_images)
    legacy_vectors = grayscale_images_to_vectors(oriented)
    shared_pairs = [normalize_glyph(image) for image in oriented]
    shared_images = [pair[0] for pair in shared_pairs]
    shared_vectors = np.stack([pair[1] for pair in shared_pairs])
    return legacy_vectors, shared_images, shared_vectors


def _select_model(
    vectors: np.ndarray,
    labels: np.ndarray,
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
    class_ids: tuple[int, ...],
    label_order: tuple[int, ...],
) -> tuple[ManualKNN, int, dict[int, float]]:
    candidates = tuple(
        k for k in K_CANDIDATES if k <= len(training_indices)
    )
    if not candidates:
        raise ValueError("Không đủ mẫu train để thử các giá trị k đã cấu hình.")
    selected_k, scores = select_k(
        vectors[training_indices],
        labels[training_indices],
        vectors[validation_indices],
        labels[validation_indices],
        class_ids,
        label_order,
        candidates,
    )
    model = ManualKNN(vectors, labels, label_order)
    return model, selected_k, scores


def run_experiment(
    data_dir: Path,
    crop_dir: Path,
    output_dir: Path,
    max_train_per_class: int,
) -> Path:
    crop_paths = sorted(crop_dir.glob("crop_*.png"))
    if not crop_paths:
        raise FileNotFoundError(f"Không tìm thấy crop_*.png trong {crop_dir}.")

    mapping, training_data = load_emnist_training(
        data_dir,
        DEFAULT_CLASS_IDS,
        max_train_per_class,
        RANDOM_STATE,
    )
    oriented = orient_emnist_images(training_data.images)
    legacy_train_vectors = grayscale_images_to_vectors(oriented)
    shared_pairs = [normalize_glyph(image) for image in oriented]
    shared_train_images = [pair[0] for pair in shared_pairs]
    shared_train_vectors = np.stack([pair[1] for pair in shared_pairs])

    train_indices, validation_indices = split_train_validation(
        training_data.labels,
        DEFAULT_CLASS_IDS,
        VALIDATION_FRACTION,
        MAX_VALIDATION_PER_CLASS,
        RANDOM_STATE,
    )
    label_order = tuple(
        sorted(DEFAULT_CLASS_IDS, key=lambda class_id: (mapping[class_id], class_id))
    )
    legacy_model, legacy_k, legacy_scores = _select_model(
        legacy_train_vectors,
        training_data.labels,
        train_indices,
        validation_indices,
        DEFAULT_CLASS_IDS,
        label_order,
    )
    shared_model, shared_k, shared_scores = _select_model(
        shared_train_vectors,
        training_data.labels,
        train_indices,
        validation_indices,
        DEFAULT_CLASS_IDS,
        label_order,
    )

    output = _reserve_output_dir(output_dir)
    crops_output = output / "crops"
    crops_output.mkdir()
    crop_rows: list[dict[str, object]] = []
    crop_diagnostics: dict[str, object] = {}
    crop_predictions: list[dict[str, object]] = []

    for index, crop_path in enumerate(crop_paths, start=1):
        with Image.open(crop_path) as source:
            crop = np.asarray(source.convert("L"), dtype=np.uint8)
        legacy_image, legacy_vector = preprocess_character_image_array_legacy(crop)
        shared_image, shared_vector = normalize_glyph(crop)
        _save_pair(
            crops_output,
            f"crop_{index:02d}",
            legacy_image,
            shared_image,
        )
        crop_predictions.append(
            {
                "crop": crop_path.name,
                "legacy_prediction": mapping[
                    legacy_model.predict_one(legacy_vector, legacy_k)[0]
                ],
                "shared_prediction": mapping[
                    shared_model.predict_one(shared_vector, shared_k)[0]
                ],
            }
        )
        crop_diagnostics[crop_path.name] = {
            "legacy": _diagnostics(legacy_image, legacy_vector),
            "shared": _diagnostics(shared_image, shared_vector),
        }

    train_output = output / "train_samples"
    train_output.mkdir()
    selected_train_indices: list[int] = []
    for class_id in DEFAULT_CLASS_IDS:
        match = np.flatnonzero(training_data.labels == class_id)
        if match.size and len(selected_train_indices) < 12:
            selected_train_indices.append(int(match[0]))
    train_diagnostics: dict[str, object] = {}
    for index, sample_index in enumerate(selected_train_indices, start=1):
        class_id = int(training_data.labels[sample_index])
        legacy_image = oriented[sample_index]
        shared_image = shared_train_images[sample_index]
        sample_name = f"sample_{index:02d}_{mapping[class_id]}"
        _save_pair(train_output, sample_name, legacy_image, shared_image)
        train_diagnostics[sample_name] = {
            "class_id": class_id,
            "character": mapping[class_id],
            "legacy": _diagnostics(
                legacy_image, legacy_train_vectors[sample_index]
            ),
            "shared": _diagnostics(
                shared_image, shared_train_vectors[sample_index]
            ),
        }

    predictions_path = output / "crop_predictions.csv"
    with predictions_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("crop", "legacy_prediction", "shared_prediction"),
        )
        writer.writeheader()
        writer.writerows(crop_predictions)

    report = {
        "experiment": "paired legacy versus shared glyph preprocessing",
        "test_set_read_or_evaluated": False,
        "random_state": RANDOM_STATE,
        "training_subset_count": int(training_data.labels.size),
        "training_subset_per_class_cap": max_train_per_class,
        "train_indices_count": int(train_indices.size),
        "validation_indices_count": int(validation_indices.size),
        "validation_fraction": VALIDATION_FRACTION,
        "max_validation_per_class": MAX_VALIDATION_PER_CLASS,
        "class_mapping": {str(key): value for key, value in mapping.items()},
        "legacy_emnist": {
            "preprocessing": "IDX transpose then grayscale vectorize / 255",
            "selected_k": legacy_k,
            "validation_macro_f1_by_k": {
                str(key): score for key, score in legacy_scores.items()
            },
        },
        "shared_glyph_pipeline": {
            "preprocessing": (
                "IDX transpose; infer polarity; crop foreground; resize to fit "
                "20x20; center on 28x28 black canvas; vectorize / 255"
            ),
            "selected_k": shared_k,
            "validation_macro_f1_by_k": {
                str(key): score for key, score in shared_scores.items()
            },
        },
        "crop_predictions": crop_predictions,
        "crop_diagnostics": crop_diagnostics,
        "labeled_train_sample_diagnostics": train_diagnostics,
        "notes": [
            "Old and new representations use the same sampled training records and stratified train/validation indices.",
            "k is selected independently for each representation using validation macro-F1.",
            "Crop labels are approximate visual notes, not verified ground truth; no crop accuracy is reported.",
            "No official test image or label file is opened.",
        ],
    }
    (output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--crop-dir",
        type=Path,
        default=RESULTS_DIR
        / "sample-01-debug-run"
        / "segmentation_debug"
        / "sample-01",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "preprocessing-comparison",
    )
    parser.add_argument(
        "--max-train-per-class",
        type=int,
        default=MAX_TRAIN_PER_CLASS,
    )
    args = parser.parse_args()
    if args.max_train_per_class < 2:
        raise SystemExit("--max-train-per-class phải tối thiểu là 2.")
    try:
        output = run_experiment(
            args.data_dir,
            args.crop_dir,
            args.output_dir,
            args.max_train_per_class,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        raise SystemExit(f"Lỗi thử nghiệm preprocessing: {error}") from error
    print(f"Comparison artifacts: {output}")
    print("Test split: not read or evaluated.")


if __name__ == "__main__":
    main()
