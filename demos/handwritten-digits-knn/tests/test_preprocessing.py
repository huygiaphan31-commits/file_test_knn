import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from preprocessing import (
    dataset_images_to_canonical,
    dataset_images_to_vectors,
    grayscale_images_to_vectors,
    normalize_glyph,
    orient_emnist_images,
    preprocess_character_image,
)


class PreprocessingTests(unittest.TestCase):
    def test_emnist_orientation_and_vector_size(self):
        image = np.zeros((1, 28, 28), dtype=np.uint8)
        image[0, 7, 13] = 255
        oriented = orient_emnist_images(image)
        self.assertEqual(oriented[0, 13, 7], 255)
        vectors = dataset_images_to_vectors(image)
        self.assertEqual(vectors.shape, (1, 784))
        self.assertEqual(float(vectors.max()), 1.0)
        self.assertEqual(vectors.dtype, np.float32)
        canonical, expected_vector = normalize_glyph(oriented[0])
        np.testing.assert_array_equal(vectors[0], expected_vector)
        np.testing.assert_array_equal(
            dataset_images_to_canonical(image)[0][0], canonical
        )

    def test_shared_glyph_pipeline_matches_emnist_and_external_polarities(self):
        ink_on_black = np.zeros((28, 28), dtype=np.uint8)
        ink_on_black[5:23, 9:17] = 220
        emnist_input = np.transpose(ink_on_black, (1, 0))[np.newaxis, ...]
        external_input = 255 - ink_on_black

        emnist_canonical, emnist_vector = dataset_images_to_canonical(emnist_input)[0]
        external_canonical, external_vector = normalize_glyph(external_input)

        np.testing.assert_array_equal(emnist_canonical, external_canonical)
        np.testing.assert_array_equal(emnist_vector, external_vector)
        self.assertEqual(emnist_canonical.shape, (28, 28))
        self.assertEqual(emnist_vector.shape, (784,))
        self.assertEqual(emnist_vector.dtype, np.float32)
        self.assertGreaterEqual(float(emnist_vector.min()), 0.0)
        self.assertLessEqual(float(emnist_vector.max()), 1.0)

        active_y, active_x = np.where(emnist_canonical > 0)
        self.assertLessEqual(int(active_x.max() - active_x.min() + 1), 20)
        self.assertLessEqual(int(active_y.max() - active_y.min() + 1), 20)
        self.assertLessEqual(abs((active_x.min() + active_x.max()) / 2 - 13.5), 1.0)
        self.assertLessEqual(abs((active_y.min() + active_y.max()) / 2 - 13.5), 1.0)

    def test_shared_glyph_pipeline_normalizes_low_contrast_glyph_intensity(self):
        low_contrast_glyph = np.full((40, 40), 240, dtype=np.uint8)
        low_contrast_glyph[10:30, 17:23] = 180

        canonical, vector = normalize_glyph(low_contrast_glyph)

        self.assertEqual(int(canonical.max()), 255)
        self.assertEqual(float(vector.max()), 1.0)
        np.testing.assert_array_equal(
            vector, grayscale_images_to_vectors(canonical[np.newaxis, ...])[0]
        )

    def test_vectorizer_checks_image_shape_and_pixel_range(self):
        with self.assertRaisesRegex(ValueError, "canonical"):
            grayscale_images_to_vectors(np.zeros((1, 28, 27), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "khoảng"):
            grayscale_images_to_vectors(np.full((1, 28, 28), 256, dtype=np.uint16))

    def test_preprocesses_single_character_to_centered_28_by_28(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "character.png"
            image = np.full((40, 40), 255, dtype=np.uint8)
            image[10:30, 17:23] = 0
            Image.fromarray(image).save(path)
            canonical, vector = preprocess_character_image(path)
            self.assertEqual(canonical.shape, (28, 28))
            self.assertEqual(vector.shape, (784,))
            self.assertEqual(vector.dtype, np.float32)
            self.assertGreaterEqual(float(vector.min()), 0.0)
            self.assertLessEqual(float(vector.max()), 1.0)
            np.testing.assert_array_equal(
                vector, grayscale_images_to_vectors(canonical[np.newaxis, ...])[0]
            )
            self.assertGreater(float(canonical.max()), 0.0)
            self.assertGreater(float(canonical.sum()), 0.0)

    def test_white_and_black_backgrounds_produce_same_canonical_character(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            white_background = np.full((40, 40), 255, dtype=np.uint8)
            white_background[10:30, 17:23] = 0
            black_background = 255 - white_background
            white_path = folder / "white-background.png"
            black_path = folder / "black-background.png"
            Image.fromarray(white_background).save(white_path)
            Image.fromarray(black_background).save(black_path)

            white_canonical, white_vector = preprocess_character_image(white_path)
            black_canonical, black_vector = preprocess_character_image(black_path)

            np.testing.assert_array_equal(white_canonical, black_canonical)
            np.testing.assert_array_equal(white_vector, black_vector)

    def test_preserves_aspect_ratio_when_resizing_user_character(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wide-character.png"
            image = np.full((40, 40), 255, dtype=np.uint8)
            image[18:22, 5:35] = 0
            Image.fromarray(image).save(path)

            canonical, _ = preprocess_character_image(path)
            active_rows, active_columns = np.where(canonical > 0)
            column_span = np.ptp(active_columns)
            row_span = np.ptp(active_rows)
            self.assertLessEqual(column_span + 1, 20)
            self.assertLessEqual(row_span + 1, 20)
            self.assertGreater(column_span, row_span)

    def test_rejects_blank_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "blank.png"
            Image.fromarray(np.full((28, 28), 255, dtype=np.uint8)).save(path)
            with self.assertRaisesRegex(ValueError, "không có nét"):
                preprocess_character_image(path)


if __name__ == "__main__":
    unittest.main()
