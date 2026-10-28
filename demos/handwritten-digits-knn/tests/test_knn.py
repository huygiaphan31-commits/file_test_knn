import unittest

import numpy as np

from knn import ManualKNN, euclidean_distance


class EuclideanDistanceTests(unittest.TestCase):
    def test_matches_textbook_formula(self):
        first = np.array([1, 2, 3])
        second = np.array([4, 6, 3])
        self.assertEqual(euclidean_distance(first, second), 5.0)

    def test_rejects_vectors_with_different_sizes(self):
        with self.assertRaises(ValueError):
            euclidean_distance(np.array([1, 2]), np.array([1]))


class ManualKNNTests(unittest.TestCase):
    def setUp(self):
        self.model = ManualKNN(
            np.array([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32),
            np.array([0, 0, 1, 1]),
            (0, 1),
        )

    def test_finds_nearest_neighbors_in_distance_order(self):
        result = self.model.find_neighbors(np.array([0.05]), 3)
        self.assertEqual(result.labels.tolist(), [0, 0, 1])
        self.assertTrue(np.all(result.distances[:-1] <= result.distances[1:]))

    def test_majority_vote_predicts_most_common_label(self):
        prediction = self.model.classify_neighbors(
            np.array([1, 0, 1]), np.array([0.1, 0.2, 0.3])
        )
        self.assertEqual(prediction, 1)

    def test_vote_tie_uses_smallest_sum_of_distances(self):
        prediction = self.model.classify_neighbors(
            np.array([1, 2, 1, 2]),
            np.array([0.2, 0.1, 0.9, 0.3]),
        )
        self.assertEqual(prediction, 2)

    def test_equal_vote_and_distance_tie_uses_fixed_label_order(self):
        model = ManualKNN(
            np.array([[0.0], [1.0]]),
            np.array([2, 1]),
            (2, 1),
        )
        prediction = model.classify_neighbors(
            np.array([1, 2]), np.array([0.5, 0.5])
        )
        self.assertEqual(prediction, 2)

    def test_rejects_invalid_k(self):
        with self.assertRaises(ValueError):
            self.model.find_neighbors(np.array([0.0]), 0)

    def test_k_larger_than_training_set_fails_without_reducing_k(self):
        with self.assertRaisesRegex(ValueError, "k phải nằm trong khoảng 1–4"):
            self.model.find_neighbors(np.array([0.0]), 1000)

    def test_predicts_single_query(self):
        prediction, _ = self.model.predict_one(np.array([0.05]), 3)
        self.assertEqual(prediction, 0)


if __name__ == "__main__":
    unittest.main()
