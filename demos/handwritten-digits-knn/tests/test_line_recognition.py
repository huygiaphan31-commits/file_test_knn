import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from line_recognition import (
    _foreground_mask,
    recognize_line,
    save_segmentation_debug,
    segment_line,
)


class FakeClassifier:
    def __init__(self, labels):
        self.labels = iter(labels)
        self.calls = 0

    def predict_one(self, query, k):
        self.calls += 1
        return next(self.labels), None


class LineRecognitionTests(unittest.TestCase):
    @staticmethod
    def make_line(background=255, positions=(8, 26, 44)):
        image = Image.new("L", (100, 40), color=background)
        draw = ImageDraw.Draw(image)
        ink = 0 if background else 255
        draw.rectangle((positions[0], 10, positions[0] + 7, 30), fill=ink)
        draw.rectangle((positions[1], 7, positions[1] + 8, 30), fill=ink)
        draw.rectangle((positions[2], 10, positions[2] + 7, 30), fill=ink)
        return np.asarray(image)

    def test_segments_characters_in_left_to_right_order(self):
        boxes = segment_line(self.make_line())
        self.assertEqual(len(boxes), 3)
        self.assertEqual([box[0] for box in boxes], sorted(box[0] for box in boxes))

    def test_segments_ten_characters_with_uneven_spacing(self):
        image = Image.new("L", (260, 50), color=245)
        draw = ImageDraw.Draw(image)
        x_position = 8
        gaps = (8, 14, 9, 18, 7, 12, 10, 16, 8)
        draw.rectangle((x_position, 10, x_position + 7, 39), fill=20)
        for gap in gaps:
            x_position += 8 + gap
            draw.rectangle((x_position, 10, x_position + 7, 39), fill=20)

        boxes = segment_line(np.asarray(image))

        self.assertEqual(len(boxes), 10)
        self.assertEqual(
            [box[0] for box in boxes],
            sorted(box[0] for box in boxes),
        )
        self.assertTrue(all(box[2] - box[0] == 8 for box in boxes))

    def test_segments_characters_over_a_shaded_paper_background(self):
        height, width = 90, 360
        vertical_gradient = np.arange(height, dtype=np.float32)[:, None]
        background = np.broadcast_to(
            242 - 1.3 * vertical_gradient,
            (height, width),
        ).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(background, mode="L")
        draw = ImageDraw.Draw(image)
        positions = tuple(range(12, 337, 36))
        for index, x_position in enumerate(positions):
            draw.line((x_position, 25, x_position, 65), fill=35, width=4)
            draw.line((x_position, 25, x_position + 10, 25), fill=35, width=4)
            if index % 2 == 0:
                draw.line((x_position, 45, x_position + 10, 45), fill=35, width=4)

        boxes = segment_line(np.asarray(image))

        self.assertEqual(len(boxes), 10)
        self.assertTrue(all(box[2] - box[0] <= 13 for box in boxes))
        self.assertTrue(all(box[3] - box[1] <= 43 for box in boxes))

    def test_uniform_gray_background_does_not_become_foreground(self):
        image = np.full((60, 100), 180, dtype=np.uint8)

        self.assertEqual(segment_line(image), ())

    def test_dark_and_light_backgrounds_have_same_segments(self):
        light_boxes = segment_line(self.make_line(background=255))
        dark_boxes = segment_line(self.make_line(background=0))
        self.assertEqual(light_boxes, dark_boxes)

    def test_detached_dot_is_grouped_with_character_body(self):
        image = Image.new("L", (70, 45), color=255)
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 17, 19, 37), fill=0)
        draw.rectangle((13, 7, 15, 9), fill=0)
        draw.rectangle((42, 14, 51, 37), fill=0)

        boxes = segment_line(np.asarray(image))
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0], (10, 7, 20, 38))

    def test_tiny_noise_away_from_characters_is_filtered(self):
        image = self.make_line().copy()
        image[1, 95] = 0
        image[2, 96] = 0

        boxes = segment_line(image)

        self.assertEqual(len(boxes), 3)
        self.assertTrue(all(box[0] < 90 for box in boxes))

    def test_repeated_thin_marks_at_image_edge_are_removed(self):
        image = Image.new("L", (120, 80), color=255)
        draw = ImageDraw.Draw(image)
        for y in range(2, 72, 12):
            draw.line((2, y, 2, y + 6), fill=0, width=1)
        draw.rectangle((35, 25, 42, 57), fill=0)

        mask = _foreground_mask(np.asarray(image))
        boxes = segment_line(np.asarray(image))

        self.assertFalse(mask[:, :4].any())
        self.assertEqual(len(boxes), 1)
        self.assertGreaterEqual(boxes[0][0], 35)

    def test_single_valid_thin_character_stroke_at_edge_is_kept(self):
        image = Image.new("L", (100, 60), color=255)
        ImageDraw.Draw(image).line((1, 12, 1, 48), fill=0, width=1)

        mask = _foreground_mask(np.asarray(image))
        boxes = segment_line(np.asarray(image))

        self.assertTrue(mask[:, :3].any())
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0][0], 1)

    def test_normal_characters_remain_segmented_with_edge_noise_removed(self):
        image = self.make_line().copy()
        for y in range(2, 37, 8):
            image[y : y + 5, 1] = 0

        boxes = segment_line(image)

        self.assertEqual(len(boxes), 3)
        self.assertTrue(all(box[0] >= 8 for box in boxes))

    def test_blank_and_edge_noise_only_images_return_no_regions(self):
        blank = np.full((80, 120), 255, dtype=np.uint8)
        noise = blank.copy()
        for y in range(2, 72, 12):
            noise[y : y + 7, 2] = 0

        self.assertEqual(segment_line(blank), ())
        self.assertEqual(segment_line(noise), ())

    def test_ruled_horizontal_line_is_removed_before_components(self):
        image = self.make_line().copy()
        image[34, :] = 0

        mask = _foreground_mask(image)
        boxes = segment_line(image)

        self.assertFalse(mask[34].any())
        self.assertEqual(len(boxes), 3)

    def test_large_gap_is_rendered_as_space_but_close_letters_are_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            close_path = folder / "close.png"
            spaced_path = folder / "spaced.png"
            Image.fromarray(self.make_line(positions=(8, 25, 42))).save(close_path)
            Image.fromarray(self.make_line(positions=(8, 50, 82))).save(spaced_path)

            close_result = recognize_line(
                close_path, FakeClassifier([10, 11, 12]), 1, {10: "A", 11: "b", 12: "3"}
            )
            spaced_result = recognize_line(
                spaced_path, FakeClassifier([10, 11, 12]), 1, {10: "A", 11: "b", 12: "3"}
            )

        self.assertEqual(close_result.text, "Ab3")
        self.assertEqual(spaced_result.text, "A b 3")

    def test_splits_wide_component_only_at_clear_vertical_valley(self):
        image = Image.new("L", (70, 40), color=255)
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 10, 22, 30), fill=0)
        draw.rectangle((31, 10, 45, 30), fill=0)
        draw.line((22, 20, 31, 20), fill=0, width=1)

        boxes = segment_line(np.asarray(image))

        self.assertEqual(len(boxes), 2)
        self.assertLessEqual(boxes[0][2], boxes[1][0])

    def test_uncertain_wide_region_is_kept_unsplit(self):
        image = Image.new("L", (60, 40), color=255)
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 40, 30), fill=0)

        boxes = segment_line(np.asarray(image))

        self.assertEqual(len(boxes), 1)

    def test_recognizes_and_joins_a_line_using_one_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "line.png"
            Image.fromarray(self.make_line()).save(path)
            model = FakeClassifier([10, 11, 12])
            mapping = {10: "A", 11: "b", 12: "3"}

            result = recognize_line(path, model, 3, mapping)

        self.assertEqual(result.text, "Ab3")
        self.assertEqual(len(result.boxes), 3)
        self.assertEqual(model.calls, 3)

    def test_blank_image_returns_empty_result_without_calling_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "blank.png"
            Image.fromarray(np.full((40, 100), 255, dtype=np.uint8)).save(path)
            model = FakeClassifier([])

            result = recognize_line(path, model, 3, {})

        self.assertEqual(result.text, "")
        self.assertEqual(result.boxes, ())
        self.assertEqual(model.calls, 0)

    def test_debug_saves_mask_annotated_regions_and_each_crop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "sample.png"
            Image.fromarray(self.make_line()).save(image_path)

            paths = save_segmentation_debug(image_path, root / "results" / "sample")

            self.assertEqual(len(paths), 5)
            self.assertTrue(paths[0].name.endswith("binary_mask.png"))
            self.assertTrue(paths[1].name.endswith("regions_numbered.png"))
            self.assertEqual(len(list(paths[0].parent.glob("crop_*.png"))), 3)

    def test_debug_csv_contains_region_geometry_and_prediction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "sample.png"
            Image.fromarray(self.make_line()).save(image_path)
            result = recognize_line(
                image_path,
                FakeClassifier([10, 11, 12]),
                1,
                {10: "A", 11: "b", 12: "3"},
            )

            paths = save_segmentation_debug(image_path, root / "debug", result)
            csv_path = paths[0].parent / "03_region_predictions.csv"
            with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 3)
        self.assertEqual([row["predicted_character"] for row in rows], ["A", "b", "3"])
        self.assertEqual(rows[0]["width"], "8")
        self.assertGreater(int(rows[0]["height"]), 0)


if __name__ == "__main__":
    unittest.main()
