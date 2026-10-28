import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from app import (
    SINGLE_CHARACTER_MODE,
    SINGLE_LINE_MODE,
    WEB_CONFIGURATION_KEY,
    WEB_K_SELECTION_MODE,
    _model_cache_state,
    load_demo_model,
    model_matches_configuration,
    prediction_key,
    predict_uploaded_image,
    refresh_demo_model,
)
from config import FIXED_EXPERIMENT_K, WEB_MAX_TRAIN_PER_CLASS


class FakeClassifier:
    def __init__(self, labels):
        self.labels = iter(labels)
        self.calls = 0

    def predict_one(self, query, k):
        self.calls += 1
        self.last_query = np.asarray(query)
        return next(self.labels), None


def encode_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class DemoAppTests(unittest.TestCase):
    def setUp(self):
        load_demo_model.clear()
        _model_cache_state.clear()

    def tearDown(self):
        load_demo_model.clear()
        _model_cache_state.clear()

    def test_single_character_mode_returns_prediction_and_normalized_glyph(self):
        image = Image.new("L", (40, 40), color=255)
        ImageDraw.Draw(image).line((8, 8, 28, 30), fill=0, width=3)
        model = FakeClassifier([10])

        result = predict_uploaded_image(
            encode_image(image),
            SINGLE_CHARACTER_MODE,
            model,
            3,
            {10: "A"},
        )

        self.assertEqual(result.text, "A")
        self.assertEqual(result.processed_image.shape, (28, 28))
        self.assertIsNone(result.annotated_image)
        self.assertEqual(model.calls, 1)
        self.assertEqual(model.last_query.shape, (784,))

    def test_line_mode_returns_ordered_regions_crops_and_labels(self):
        image = Image.new("L", (100, 45), color=255)
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 18, 34), fill=0)
        draw.rectangle((42, 10, 50, 34), fill=0)
        model = FakeClassifier([10, 11])

        result = predict_uploaded_image(
            encode_image(image),
            SINGLE_LINE_MODE,
            model,
            3,
            {10: "A", 11: "b"},
        )

        self.assertEqual(result.text, "A b")
        self.assertEqual(len(result.crops), 2)
        self.assertEqual(result.region_predictions, ("A", "b"))
        self.assertIsNotNone(result.annotated_image)
        self.assertEqual(model.calls, 2)

    def test_blank_line_returns_empty_prediction_without_model_calls(self):
        image = Image.new("L", (80, 40), color=255)
        model = FakeClassifier([])

        result = predict_uploaded_image(
            encode_image(image),
            SINGLE_LINE_MODE,
            model,
            3,
            {},
        )

        self.assertEqual(result.text, "")
        self.assertEqual(result.crops, ())
        self.assertEqual(model.calls, 0)

    def test_unknown_mode_reports_error(self):
        with self.assertRaisesRegex(ValueError, "Chế độ ảnh không hợp lệ"):
            predict_uploaded_image(b"unused", "không hợp lệ", FakeClassifier([]), 1, {})

    def test_web_exposes_only_fixed_500_sample_k101_configuration(self):
        self.assertEqual(WEB_MAX_TRAIN_PER_CLASS, 500)
        self.assertEqual(WEB_K_SELECTION_MODE, "fixed_101")
        self.assertEqual(WEB_CONFIGURATION_KEY, (500, "fixed_101"))
        self.assertEqual(FIXED_EXPERIMENT_K, 101)

    def test_load_demo_model_uses_fixed_500_sample_configuration_and_k101(self):
        training_data = SimpleNamespace(labels=np.zeros(4, dtype=np.int64))
        model = object()
        scores = {101: 0.9}
        with (
            patch("app.load_emnist_training", return_value=({}, training_data)) as load,
            patch(
                "app.fit_classifier",
                return_value=(
                    model,
                    101,
                    scores,
                    np.array([0, 1]),
                    np.array([2, 3]),
                    (101,),
                ),
            ) as patch_fit_classifier,
        ):
            model_info = load_demo_model("test-data")

        self.assertEqual(load.call_args.args[2], 500)
        self.assertEqual(model_info.selected_k, 101)
        self.assertEqual(model_info.validation_macro_f1, 0.9)
        self.assertEqual(model_info.training_count, 4)
        self.assertEqual(model_info.configuration_key, (500, "fixed_101"))
        self.assertTrue(model_matches_configuration(model_info, WEB_CONFIGURATION_KEY))
        self.assertEqual(
            patch_fit_classifier.call_args.kwargs["k_candidates"],
            (101,),
        )

    def test_load_demo_model_rejects_other_configurations(self):
        with self.assertRaisesRegex(ValueError, "Web chỉ hỗ trợ"):
            load_demo_model("test-data", 200, "validation")

    def test_cache_reuses_only_the_single_web_configuration(self):
        training_data = SimpleNamespace(labels=np.zeros(2, dtype=np.int64))
        scores = {101: 0.9}
        with (
            patch("app.load_emnist_training", return_value=({}, training_data)) as load,
            patch(
                "app.fit_classifier",
                return_value=(
                    object(),
                    101,
                    scores,
                    np.array([0]),
                    np.array([1]),
                    (101,),
                ),
            ),
        ):
            first = load_demo_model("cache-isolation-test")
            same_configuration = load_demo_model("cache-isolation-test")

        self.assertIs(first, same_configuration)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(first.configuration_key, WEB_CONFIGURATION_KEY)
        self.assertTrue(model_matches_configuration(first, WEB_CONFIGURATION_KEY))
        self.assertFalse(model_matches_configuration(first, (200, "validation")))
        stale_model = first.__class__(
            model=first.model,
            selected_k=9,
            mapping=first.mapping,
            training_count=first.training_count,
            validation_count=first.validation_count,
            validation_macro_f1=first.validation_macro_f1,
            max_train_per_class=first.max_train_per_class,
            k_selection_mode=first.k_selection_mode,
        )
        self.assertFalse(model_matches_configuration(stale_model, WEB_CONFIGURATION_KEY))

    def test_prediction_key_changes_when_image_content_changes(self):
        configuration = WEB_CONFIGURATION_KEY
        first = prediction_key(b"image bytes one", SINGLE_LINE_MODE, configuration)
        same_image = prediction_key(b"image bytes one", SINGLE_LINE_MODE, configuration)
        changed_image = prediction_key(b"image bytes two", SINGLE_LINE_MODE, configuration)

        self.assertEqual(first, same_image)
        self.assertNotEqual(first, changed_image)

    def test_refresh_clears_cached_model_and_advances_shared_generation(self):
        training_data = SimpleNamespace(labels=np.zeros(4, dtype=np.int64))
        scores = {101: 0.9}

        def fitted_model():
            return (
                object(),
                101,
                scores,
                np.array([0, 1]),
                np.array([2, 3]),
                (101,),
            )

        with (
            patch("app.load_emnist_training", return_value=({}, training_data)) as load,
            patch("app.fit_classifier", side_effect=[fitted_model(), fitted_model()]),
        ):
            first = load_demo_model("refresh-test")
            generation_before = _model_cache_state()["generation"]
            refreshed = refresh_demo_model("refresh-test")

        self.assertIsNot(first, refreshed)
        self.assertEqual(load.call_count, 2)
        self.assertEqual(generation_before, 0)
        self.assertEqual(_model_cache_state()["generation"], 1)
        self.assertEqual(refreshed.training_count, 4)
        self.assertEqual(refreshed.selected_k, 101)


if __name__ == "__main__":
    unittest.main()
