import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "eight-characters-diagnosis"
    / "evaluate_labeled_contrast.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_labeled_contrast", SCRIPT_PATH)
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


class ContrastEvaluationTests(unittest.TestCase):
    def test_macro_f1_and_confusion_include_predictions_outside_ground_truth(self):
        metrics = EVALUATOR._metrics(["a", "b"], ["x", "b"])

        self.assertEqual(metrics["correct"], 1)
        self.assertEqual(metrics["total"], 2)
        self.assertEqual(metrics["macro_f1_labels"], ["a", "b", "x"])
        self.assertAlmostEqual(
            metrics["macro_f1_over_union_of_true_and_predicted_classes"], 1 / 3
        )
        self.assertEqual(metrics["confusion_matrix"]["a"]["x"], 1)
        self.assertEqual(metrics["per_class"]["x"]["false_positive"], 1)

    def test_manifest_requires_confirmed_unseen_independent_holdout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "sample.png"
            image_path.write_bytes(b"fixture")
            row = {
                "image": "sample.png",
                "ground_truth_characters": "aB3",
                "ground_truth_source": "confirmed before prediction",
                "evaluation_role": "independent_holdout",
                "label_confirmed_before_prediction": "yes",
                "used_for_method_selection": "no",
                "used_in_prior_diagnostics": "no",
            }

            validated = EVALUATOR._validate_manifest([row], root)

            self.assertEqual(validated[0][1], image_path.resolve())
            self.assertEqual(validated[0][2], "aB3")

            row["used_for_method_selection"] = "yes"
            with self.assertRaisesRegex(ValueError, "independent holdout"):
                EVALUATOR._validate_manifest([row], root)

    def test_manifest_rejects_path_escape_and_region_label_spaces(self):
        base = {
            "image": "..\\outside.png",
            "ground_truth_characters": "ab",
            "ground_truth_source": "confirmed",
            "evaluation_role": "pilot",
            "label_confirmed_before_prediction": "yes",
            "used_for_method_selection": "yes",
            "used_in_prior_diagnostics": "yes",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "escapes"):
                EVALUATOR._validate_manifest([base], root)

            image_path = root / "sample.png"
            image_path.write_bytes(b"fixture")
            base["image"] = "sample.png"
            base["ground_truth_characters"] = "a b"
            with self.assertRaisesRegex(ValueError, "omitting them"):
                EVALUATOR._validate_manifest([base], root)


if __name__ == "__main__":
    unittest.main()
