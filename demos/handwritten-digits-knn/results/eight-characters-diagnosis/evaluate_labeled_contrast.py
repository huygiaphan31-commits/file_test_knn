"""Evaluate current and peak-normalized queries on labeled local images."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DEMO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DEMO_DIR))

from app import load_demo_model
from config import DATA_DIR
from line_recognition import _padded_crop, load_grayscale_image, segment_line
from preprocessing import preprocess_character_image_array


HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "labeled_evaluation_manifest.csv"
OUTPUT_ROOT = HERE / "labeled_contrast_evaluation"
K = 9
EVALUATION_ROLES = {"pilot", "previously_used_comparison", "independent_holdout"}
ALLOWED_CHARACTERS = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")


def _read_manifest() -> list[dict[str, str]]:
    with MANIFEST_PATH.open(encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    required = {
        "image",
        "ground_truth_characters",
        "ground_truth_source",
        "evaluation_role",
        "label_confirmed_before_prediction",
        "used_for_method_selection",
        "used_in_prior_diagnostics",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain columns: {sorted(required)}")
    return rows


def _validate_manifest(
    rows: list[dict[str, str]],
    demo_dir: Path = DEMO_DIR,
) -> list[tuple[dict[str, str], Path, str]]:
    validated = []
    seen_images: set[Path] = set()
    demo_root = demo_dir.resolve()
    for row_number, row in enumerate(rows, start=2):
        image_value = row.get("image", "").strip()
        ground_truth = row.get("ground_truth_characters", "").strip()
        source = row.get("ground_truth_source", "").strip()
        role = row.get("evaluation_role", "").strip()
        if not image_value or not ground_truth or not source:
            raise ValueError(
                f"Manifest row {row_number} must include image, ground truth, "
                "and ground-truth source."
            )
        if role not in EVALUATION_ROLES:
            raise ValueError(
                f"Manifest row {row_number} has invalid evaluation_role {role!r}."
            )
        if any(character.isspace() for character in ground_truth):
            raise ValueError(
                f"Manifest row {row_number}: encode line spaces by omitting them "
                "from ground_truth_characters."
            )
        invalid_characters = set(ground_truth) - ALLOWED_CHARACTERS
        if invalid_characters:
            raise ValueError(
                f"Manifest row {row_number} has unsupported labels: "
                f"{sorted(invalid_characters)!r}."
            )

        image_path_value = Path(image_value)
        if image_path_value.is_absolute():
            raise ValueError(
                f"Manifest row {row_number}: image path must be relative to the demo."
            )
        image_path = (demo_root / image_path_value).resolve()
        if not image_path.is_relative_to(demo_root):
            raise ValueError(
                f"Manifest row {row_number}: image path escapes the demo directory."
            )
        if image_path in seen_images:
            raise ValueError(
                f"Manifest contains duplicate image entries: {image_value!r}."
            )
        seen_images.add(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        if role == "independent_holdout":
            holdout_requirements = {
                "label_confirmed_before_prediction": "yes",
                "used_for_method_selection": "no",
                "used_in_prior_diagnostics": "no",
            }
            for field, expected_value in holdout_requirements.items():
                if row.get(field, "").strip().lower() != expected_value:
                    raise ValueError(
                        f"Manifest row {row_number} is not eligible as an "
                        f"independent holdout: {field} must be {expected_value!r}."
                    )
        validated.append((row, image_path, ground_truth))
    if not validated:
        raise ValueError("Manifest contains no labeled images.")
    return validated


def _expected_rank(model_info, vector: np.ndarray, expected_id: int) -> int:
    differences = model_info.model.training_vectors - vector
    distances = np.sqrt(
        np.sum(differences * differences, axis=1, dtype=np.float64)
    )
    expected_distance = float(
        distances[model_info.model.training_labels == expected_id].min()
    )
    return int(np.count_nonzero(distances < expected_distance)) + 1


def _predict(model_info, vector: np.ndarray) -> tuple[str, list[str], list[float]]:
    predicted, neighbors = model_info.model.predict_one(vector, K)
    return (
        model_info.mapping[int(predicted)],
        [model_info.mapping[int(label)] for label in neighbors.labels],
        [float(distance) for distance in neighbors.distances],
    )


def _metrics(
    expected: list[str],
    predicted: list[str],
) -> dict[str, object]:
    if not expected or len(expected) != len(predicted):
        raise ValueError("Expected and predicted labels must have the same nonzero size.")
    labels = sorted(set(expected) | set(predicted))
    per_class = {}
    f1_values = []
    confusion = {actual: {guess: 0 for guess in labels} for actual in labels}
    for actual, guess in zip(expected, predicted):
        confusion[actual][guess] += 1
    for label in labels:
        true_positive = sum(
            actual == label and guess == label
            for actual, guess in zip(expected, predicted)
        )
        false_positive = sum(
            actual != label and guess == label
            for actual, guess in zip(expected, predicted)
        )
        false_negative = sum(
            actual == label and guess != label
            for actual, guess in zip(expected, predicted)
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        per_class[label] = {
            "support": sum(actual == label for actual in expected),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "correct": sum(actual == guess for actual, guess in zip(expected, predicted)),
        "total": len(expected),
        "accuracy": sum(actual == guess for actual, guess in zip(expected, predicted))
        / len(expected),
        "macro_f1_over_union_of_true_and_predicted_classes": sum(f1_values)
        / len(f1_values),
        "macro_f1_labels": labels,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def _new_output_dir() -> Path:
    output_dir = OUTPUT_ROOT / datetime.now().strftime("run-%Y%m%d-%H%M%S-%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def main() -> None:
    manifest = _validate_manifest(_read_manifest())
    prepared_images = []
    for sample, image_path, expected_text in manifest:
        image = load_grayscale_image(image_path)
        boxes = segment_line(image)
        if len(boxes) != len(expected_text):
            raise ValueError(
                f"{sample['image']}: expected {len(expected_text)} labeled regions, "
                f"but segmentation found {len(boxes)}."
            )
        prepared_images.append((sample, image_path, expected_text, image, boxes))

    model_info = load_demo_model(str(DATA_DIR), 500, "fixed_9")
    if model_info.training_count != 31_000 or model_info.selected_k != K:
        raise ValueError("Expected model B: 31,000 training samples and k=9.")
    if set(model_info.mapping.values()) != ALLOWED_CHARACTERS:
        raise ValueError("Loaded model mapping does not contain the expected 62 labels.")
    OUTPUT_DIR = _new_output_dir()

    all_expected: list[str] = []
    all_current: list[str] = []
    all_peak_normalized: list[str] = []
    independent_expected: list[str] = []
    independent_current: list[str] = []
    independent_peak_normalized: list[str] = []
    image_reports = []
    char_rows = []

    for image_number, (sample, image_path, expected_text, image, boxes) in enumerate(
        prepared_images, start=1
    ):
        relative_image = Path(sample["image"])
        per_image_expected = list(expected_text)
        per_image_current = []
        per_image_contrast = []
        image_dir = OUTPUT_DIR / f"image-{image_number:02d}-{image_path.stem}"
        image_dir.mkdir(parents=True, exist_ok=True)
        preview_rows = []

        for index, (expected, box) in enumerate(
            zip(per_image_expected, boxes), start=1
        ):
            crop = _padded_crop(image, box)
            current_image, current_vector = preprocess_character_image_array(crop)
            peak = float(current_vector.max())
            if peak <= 0:
                raise ValueError(f"{relative_image}, region {index}: blank glyph.")
            contrast_vector = np.clip(current_vector / peak, 0, 1).astype(np.float32)
            contrast_image = np.rint(contrast_vector.reshape(28, 28) * 255).astype(
                np.uint8
            )

            current_pred, current_neighbors, current_distances = _predict(
                model_info, current_vector
            )
            contrast_pred, contrast_neighbors, contrast_distances = _predict(
                model_info, contrast_vector
            )
            expected_id = next(
                class_id
                for class_id, character in model_info.mapping.items()
                if character == expected
            )

            per_image_current.append(current_pred)
            per_image_contrast.append(contrast_pred)

            region_dir = image_dir / f"region_{index:02d}"
            region_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(crop, mode="L").save(region_dir / "crop.png")
            Image.fromarray(current_image, mode="L").save(
                region_dir / "current_28x28.png"
            )
            Image.fromarray(contrast_image, mode="L").save(
                region_dir / "peak_normalized_28x28.png"
            )
            char_rows.append(
                {
                    "image": relative_image.as_posix(),
                    "evaluation_role": sample["evaluation_role"],
                    "ground_truth_source": sample["ground_truth_source"],
                    "region": index,
                    "expected": expected,
                    "box_xyxy": json.dumps(box),
                    "current_prediction": current_pred,
                    "current_expected_class_rank": _expected_rank(
                        model_info, current_vector, expected_id
                    ),
                    "current_neighbors": json.dumps(current_neighbors),
                    "current_distances": json.dumps(current_distances),
                    "peak_normalized_prediction": contrast_pred,
                    "peak_normalized_expected_class_rank": _expected_rank(
                        model_info, contrast_vector, expected_id
                    ),
                    "peak_normalized_neighbors": json.dumps(contrast_neighbors),
                    "peak_normalized_distances": json.dumps(contrast_distances),
                    "input_peak": peak,
                    "current_image": (
                        f"{image_dir.name}/region_{index:02d}/current_28x28.png"
                    ),
                    "peak_normalized_image": (
                        f"{image_dir.name}/region_{index:02d}/"
                        "peak_normalized_28x28.png"
                    ),
                }
            )

            crop_preview = Image.fromarray(crop, mode="L").convert("RGB")
            crop_preview.thumbnail((130, 90))
            current_preview = Image.fromarray(current_image, mode="L").resize(
                (84, 84), Image.Resampling.NEAREST
            ).convert("RGB")
            contrast_preview = Image.fromarray(contrast_image, mode="L").resize(
                (84, 84), Image.Resampling.NEAREST
            ).convert("RGB")
            preview_rows.append(
                (expected, crop_preview, current_preview, contrast_preview)
            )

        current_metrics = _metrics(per_image_expected, per_image_current)
        contrast_metrics = _metrics(per_image_expected, per_image_contrast)
        image_reports.append(
            {
                "image": relative_image.as_posix(),
                "evaluation_role": sample["evaluation_role"],
                "ground_truth_source": sample["ground_truth_source"],
                "ground_truth_characters": expected_text,
                "regions": len(boxes),
                "current": current_metrics,
                "peak_normalized": contrast_metrics,
            }
        )
        all_expected.extend(per_image_expected)
        all_current.extend(per_image_current)
        all_peak_normalized.extend(per_image_contrast)
        if sample["evaluation_role"] == "independent_holdout":
            independent_expected.extend(per_image_expected)
            independent_current.extend(per_image_current)
            independent_peak_normalized.extend(per_image_contrast)

        cell_width, cell_height = 150, 120
        sheet = Image.new(
            "RGB",
            (cell_width * 3, 28 + cell_height * len(preview_rows)),
            (245, 245, 245),
        )
        draw = ImageDraw.Draw(sheet)
        for col, title in enumerate(("Crop", "Current", "Peak normalized")):
            draw.text((col * cell_width + 4, 4), title, fill=(0, 0, 0))
        for row, (expected, crop_preview, current_preview, contrast_preview) in enumerate(
            preview_rows
        ):
            y = 28 + row * cell_height
            draw.text((4, y + 3), f"{row + 1}: {expected}", fill=(0, 0, 0))
            for col, preview in enumerate(
                (crop_preview, current_preview, contrast_preview)
            ):
                sheet.paste(preview, (col * cell_width + 5, y + 23))
        sheet.save(image_dir / "comparison.png")

    overall = {
        "current": _metrics(all_expected, all_current),
        "peak_normalized": _metrics(all_expected, all_peak_normalized),
    }
    independent_report = None
    if independent_expected:
        independent_report = {
            "current": _metrics(independent_expected, independent_current),
            "peak_normalized": _metrics(
                independent_expected, independent_peak_normalized
            ),
        }
    report = {
        "model": {
            "training_count": model_info.training_count,
            "max_train_per_class": model_info.max_train_per_class,
            "k": model_info.selected_k,
            "validation_macro_f1": model_info.validation_macro_f1,
            "test_idx_read": False,
        },
        "contrast_method": (
            "After current 28x28 glyph normalization, divide the input vector "
            "by that glyph's maximum intensity; train vectors are unchanged."
        ),
        "macro_f1_definition": (
            "Macro-F1 is averaged over the union of labels present in ground truth "
            "and predictions. With few samples and one or fewer examples/class, "
            "it is descriptive only, not a generalization estimate."
        ),
        "images": image_reports,
        "pooled_all_labeled_images": overall,
        "independent_holdout": independent_report,
        "limitations": [
            "The eight-character image was used as the pilot for this contrast method.",
            "sample-01.png has prior diagnostic/experiment results and is not a blind holdout.",
            "sample-02.png has no confirmed ground truth and is excluded.",
            "No generalization improvement can be claimed without independent holdout images.",
        ],
        "artifacts": {
            "per_character_csv": "per_character.csv",
            "confusion_matrices_csv": "confusion_matrices.csv",
        },
    }
    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT_DIR / "per_character.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(char_rows[0]))
        writer.writeheader()
        writer.writerows(char_rows)

    with (OUTPUT_DIR / "confusion_matrices.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ("scope", "method", "ground_truth", "predicted", "count")
        )
        for scope, metrics_by_method in (
            ("pooled_all_labeled_images", overall),
            ("independent_holdout", independent_report),
        ):
            if metrics_by_method is None:
                continue
            for method, metrics in metrics_by_method.items():
                for actual, predictions_by_label in metrics["confusion_matrix"].items():
                    for predicted, count in predictions_by_label.items():
                        writer.writerow((scope, method, actual, predicted, count))

    print(
        f"Model: n={model_info.training_count}, k={K}, "
        f"validation macro-F1={model_info.validation_macro_f1:.6f}"
    )
    for image_report in image_reports:
        print(
            f"{image_report['image']} ({image_report['evaluation_role']}): "
            f"current={image_report['current']['correct']}/"
            f"{image_report['current']['total']} "
            f"(macro-F1={image_report['current']['macro_f1_over_union_of_true_and_predicted_classes']:.4f}), "
            f"peak={image_report['peak_normalized']['correct']}/"
            f"{image_report['peak_normalized']['total']} "
            f"(macro-F1={image_report['peak_normalized']['macro_f1_over_union_of_true_and_predicted_classes']:.4f})"
        )
    print(
        f"Pooled: current={overall['current']['correct']}/{overall['current']['total']}, "
        f"peak={overall['peak_normalized']['correct']}/"
        f"{overall['peak_normalized']['total']}"
    )
    print(f"Independent holdout: {len(independent_expected)} labeled characters")
    print(f"Artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
