import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from main import fit_classifier


class FitClassifierTests(unittest.TestCase):
    def test_fixed_k_above_available_training_samples_fails_explicitly(self):
        training_data = SimpleNamespace(
            images=np.zeros((3, 28, 28), dtype=np.uint8),
            labels=np.array([0, 1, 0], dtype=np.int64),
        )

        with (
            patch(
                "main.dataset_images_to_vectors",
                return_value=np.zeros((3, 784), dtype=np.float32),
            ),
            patch(
                "main.split_train_validation",
                return_value=(
                    np.array([0, 1], dtype=np.int64),
                    np.array([2], dtype=np.int64),
                ),
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "k=1000; còn 2 mẫu. k không được tự động giảm",
            ):
                fit_classifier(
                    training_data,
                    (0, 1),
                    {0: "0", 1: "1"},
                    k_candidates=(1000,),
                )


if __name__ == "__main__":
    unittest.main()
