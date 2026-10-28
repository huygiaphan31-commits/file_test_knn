import unittest

import numpy as np

from evaluation import (
    confusion_matrix,
    macro_f1,
    select_k,
    split_train_validation,
)


class EvaluationTests(unittest.TestCase):
    def test_split_keeps_each_class_in_train_and_validation(self):
        labels = np.array([0] * 10 + [1] * 10)
        train, validation = split_train_validation(
            labels, (0, 1), 0.2, 2, 42
        )
        self.assertEqual(len(validation), 4)
        self.assertEqual(set(labels[train]), {0, 1})
        self.assertEqual(set(labels[validation]), {0, 1})
        self.assertTrue(set(train).isdisjoint(validation))

    def test_confusion_matrix_and_macro_f1(self):
        matrix = confusion_matrix(
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 1]),
            (0, 1),
        )
        self.assertEqual(matrix.tolist(), [[1, 1], [0, 2]])
        self.assertAlmostEqual(macro_f1(
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 1]),
            (0, 1),
        ), 11 / 15)

    def test_selects_k_using_validation_only(self):
        training_vectors = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32)
        training_labels = np.array([0, 0, 1, 1])
        validation_vectors = np.array([[0.05], [1.05]], dtype=np.float32)
        validation_labels = np.array([0, 1])
        best_k, scores = select_k(
            training_vectors,
            training_labels,
            validation_vectors,
            validation_labels,
            (0, 1),
            (0, 1),
            (1, 3),
        )
        self.assertEqual(best_k, 1)
        self.assertEqual(scores, {1: 1.0, 3: 1.0})


if __name__ == "__main__":
    unittest.main()
