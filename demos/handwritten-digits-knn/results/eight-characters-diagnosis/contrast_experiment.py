"""Compare current glyph vectors with per-glyph peak normalization."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DEMO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DEMO_DIR))

from app import load_demo_model
from config import DATA_DIR
from line_recognition import _padded_crop, load_grayscale_image, segment_line
from preprocessing import preprocess_character_image_array


IMAGE_PATH = DEMO_DIR / "data" / "my_samples" / "eight-characters.png"
OUTPUT_DIR = Path(__file__).resolve().parent / "contrast-experiment"
EXPECTED = "aAbBcCdD"
K = 9


def _class_id_for_character(mapping: dict[int, str], character: str) -> int:
    return next(class_id for class_id, value in mapping.items() if value == character)


def _diagnose_variant(
    model_info,
    vector: np.ndarray,
) -> tuple[str, list[str], list[float], int, float]:
    prediction, neighbors = model_info.model.predict_one(vector, K)
    return (
        model_info.mapping[int(prediction)],
        [model_info.mapping[int(label)] for label in neighbors.labels],
        [float(distance) for distance in neighbors.distances],
        int(neighbors.indices[0]),
        float(neighbors.distances[0]),
    )


def _expected_class_rank(model_info, vector: np.ndarray, expected_id: int) -> int:
    differences = model_info.model.training_vectors - vector
    distances = np.sqrt(
        np.sum(differences * differences, axis=1, dtype=np.float64)
    )
    expected_distance = float(distances[model_info.model.training_labels == expected_id].min())
    return int(np.count_nonzero(distances < expected_distance)) + 1


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image = load_grayscale_image(IMAGE_PATH)
    boxes = segment_line(image)
    if len(boxes) != len(EXPECTED):
        raise ValueError(
            f"Expected {len(EXPECTED)} regions for {EXPECTED!r}; found {len(boxes)}."
        )

    model_info = load_demo_model(str(DATA_DIR), 500, "fixed_9")
    if model_info.training_count != 31_000 or model_info.selected_k != K:
        raise ValueError(
            "Loaded model does not match the requested 31,000-sample, k=9 setup."
        )

    records: list[dict[str, object]] = []
    preview_rows: list[tuple[Image.Image, Image.Image, Image.Image]] = []
    predictions: dict[str, list[str]] = {"current": [], "peak_normalized": []}

    for index, (box, expected) in enumerate(zip(boxes, EXPECTED), start=1):
        crop = _padded_crop(image, box)
        current_image, current_vector = preprocess_character_image_array(crop)
        peak = float(current_vector.max())
        if peak <= 0:
            raise ValueError(f"Region {index} has no foreground intensity.")

        contrast_vector = np.clip(current_vector / peak, 0.0, 1.0).astype(np.float32)
        contrast_image = np.rint(contrast_vector.reshape(28, 28) * 255).astype(
            np.uint8
        )

        if current_image.shape != contrast_image.shape:
            raise ValueError(f"Region {index} changed shape during contrast test.")
        if not np.array_equal(current_image == 0, contrast_image == 0):
            raise ValueError(f"Region {index} changed its zero-background mask.")
        if not np.isfinite(contrast_vector).all():
            raise ValueError(f"Region {index} produced a non-finite query vector.")

        current_result = _diagnose_variant(model_info, current_vector)
        contrast_result = _diagnose_variant(model_info, contrast_vector)
        predictions["current"].append(current_result[0])
        predictions["peak_normalized"].append(contrast_result[0])
        expected_id = _class_id_for_character(model_info.mapping, expected)

        region_dir = OUTPUT_DIR / f"region_{index:02d}"
        region_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(crop, mode="L").save(region_dir / "crop.png")
        Image.fromarray(current_image, mode="L").save(region_dir / "current_28x28.png")
        Image.fromarray(contrast_image, mode="L").save(
            region_dir / "peak_normalized_28x28.png"
        )

        records.append(
            {
                "region": index,
                "expected": expected,
                "box_xyxy": list(box),
                "crop_size_wh": [crop.shape[1], crop.shape[0]],
                "input_peak": peak,
                "contrast_gain": 1.0 / peak,
                "current_prediction": current_result[0],
                "current_neighbors": current_result[1],
                "current_distances": current_result[2],
                "current_expected_class_rank": _expected_class_rank(
                    model_info, current_vector, expected_id
                ),
                "peak_normalized_prediction": contrast_result[0],
                "peak_normalized_neighbors": contrast_result[1],
                "peak_normalized_distances": contrast_result[2],
                "peak_normalized_expected_class_rank": _expected_class_rank(
                    model_info, contrast_vector, expected_id
                ),
                "current_nonzero_pixels": int(np.count_nonzero(current_image)),
                "contrast_nonzero_pixels": int(np.count_nonzero(contrast_image)),
                "current_pixels_at_least_10pct_peak": int(
                    np.count_nonzero(current_vector >= peak * 0.1)
                ),
                "contrast_pixels_at_least_10pct_peak": int(
                    np.count_nonzero(contrast_vector >= 0.1)
                ),
                "current_file": f"region_{index:02d}/current_28x28.png",
                "peak_normalized_file": f"region_{index:02d}/peak_normalized_28x28.png",
            }
        )

        crop_preview = Image.fromarray(crop, mode="L").convert("RGB")
        crop_preview.thumbnail((140, 100))
        current_preview = Image.fromarray(current_image, mode="L").resize(
            (112, 112), Image.Resampling.NEAREST
        ).convert("RGB")
        contrast_preview = Image.fromarray(contrast_image, mode="L").resize(
            (112, 112), Image.Resampling.NEAREST
        ).convert("RGB")
        preview_rows.append((crop_preview, current_preview, contrast_preview))

    fields = list(records[0])
    with (OUTPUT_DIR / "comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    report = {
        "image": str(IMAGE_PATH.relative_to(DEMO_DIR)),
        "image_size_wh": [image.shape[1], image.shape[0]],
        "expected": EXPECTED,
        "boxes_xyxy_left_to_right": [list(box) for box in boxes],
        "model": {
            "training_count": model_info.training_count,
            "max_train_per_class": model_info.max_train_per_class,
            "k": model_info.selected_k,
            "validation_macro_f1": model_info.validation_macro_f1,
            "test_idx_read": False,
        },
        "contrast_method": (
            "After existing 28x28 normalization, divide each glyph vector by "
            "its own maximum intensity; no geometry, polarity, or model change."
        ),
        "predictions": {
            "current": "".join(predictions["current"]),
            "peak_normalized": "".join(predictions["peak_normalized"]),
        },
        "characters_correct": {
            "current": sum(
                actual == expected
                for actual, expected in zip(predictions["current"], EXPECTED)
            ),
            "peak_normalized": sum(
                actual == expected
                for actual, expected in zip(predictions["peak_normalized"], EXPECTED)
            ),
            "total": len(EXPECTED),
        },
        "regions": records,
        "interpretation_limit": (
            "This is one image and cannot establish generalization improvement."
        ),
    }
    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cell_width, cell_height = 180, 145
    sheet = Image.new(
        "RGB", (cell_width * 3, cell_height * len(preview_rows)), (245, 245, 245)
    )
    draw = ImageDraw.Draw(sheet)
    headings = ("Crop", "Current 28x28", "Peak normalized")
    for col, heading in enumerate(headings):
        draw.text((col * cell_width + 6, 3), heading, fill=(0, 0, 0))
    for row, previews in enumerate(preview_rows):
        y = row * cell_height
        draw.text((6, y + 20), f"{row + 1}: {EXPECTED[row]}", fill=(0, 0, 0))
        for col, preview in enumerate(previews):
            sheet.paste(preview, (col * cell_width + 8, y + 38))
    sheet.save(OUTPUT_DIR / "comparison.png")

    print(f"Image: {IMAGE_PATH}")
    print(
        f"Model: {model_info.training_count} train samples, k={model_info.selected_k}, "
        f"validation macro-F1={model_info.validation_macro_f1:.6f}"
    )
    print(f"Expected:         {EXPECTED}")
    print(f"Current:          {report['predictions']['current']}")
    print(f"Peak normalized:  {report['predictions']['peak_normalized']}")
    print(
        "Correct on this image: "
        f"{report['characters_correct']['current']}/8 -> "
        f"{report['characters_correct']['peak_normalized']}/8"
    )
    print(f"Artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
