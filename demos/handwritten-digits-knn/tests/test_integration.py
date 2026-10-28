import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tests.helpers import write_demo_fixture


DEMO_DIR = Path(__file__).resolve().parents[1]


class DemoIntegrationTests(unittest.TestCase):
    def test_cli_runs_on_synthetic_idx_and_writes_real_run_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "EMNIST"
            results_dir = root / "results"
            write_demo_fixture(data_dir)
            query_path = root / "query.png"
            query = np.full((28, 28), 255, dtype=np.uint8)
            query[4:12, 4:12] = 0
            Image.fromarray(query).save(query_path)
            command = [
                sys.executable,
                str(DEMO_DIR / "main.py"),
                "--data-dir",
                str(data_dir),
                "--classes",
                "0,1",
                "--max-train-per-class",
                "10",
                "--max-test-per-class",
                "5",
                "--samples",
                "5",
                "--predict-image",
                str(query_path),
                "--debug-segmentation",
                "--results-dir",
                str(results_dir),
            ]
            completed = subprocess.run(
                command,
                cwd=DEMO_DIR,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            payload = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset"], "EMNIST ByClass")
            self.assertEqual(payload["test_metrics"]["class_ids"], [0, 1])
            self.assertEqual(payload["split"]["official_test_sampled_count"], 10)
            self.assertEqual(len(payload["sample_predictions"]), 5)
            for filename in (
                "metrics.json",
                "test_predictions.csv",
                "confusion_matrix.csv",
                "confusion_matrix.png",
                "test_examples.png",
                "query_prediction.png",
            ):
                self.assertTrue((results_dir / filename).is_file(), filename)
            debug_dir = results_dir / "segmentation_debug" / "query"
            self.assertTrue((debug_dir / "01_binary_mask.png").is_file())
            self.assertTrue((debug_dir / "02_regions_numbered.png").is_file())
            self.assertGreaterEqual(len(list(debug_dir.glob("crop_*.png"))), 1)
            self.assertIn("predicted character:", completed.stdout)

    def test_prediction_only_accepts_multiple_images_without_test_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "EMNIST"
            results_dir = root / "results"
            write_demo_fixture(data_dir)
            (data_dir / "emnist-byclass-test-images-idx3-ubyte.gz").unlink()
            (data_dir / "emnist-byclass-test-labels-idx1-ubyte.gz").unlink()
            image_paths = []
            for index in range(2):
                image_path = root / f"query-{index}.png"
                image = np.full((28, 28), 255, dtype=np.uint8)
                image[5:22, 8 + index * 3 : 14 + index * 3] = 0
                Image.fromarray(image).save(image_path)
                image_paths.append(str(image_path))

            command = [
                sys.executable,
                str(DEMO_DIR / "main.py"),
                "--data-dir",
                str(data_dir),
                "--classes",
                "0,1",
                "--max-train-per-class",
                "10",
                "--skip-evaluation",
                "--predict-images",
                *image_paths,
                "--results-dir",
                str(results_dir),
            ]
            completed = subprocess.run(
                command,
                cwd=DEMO_DIR,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(completed.stdout.count("predicted character:"), 2)
            self.assertFalse((results_dir / "metrics.json").exists())
            self.assertTrue((results_dir / "query_prediction_01.png").is_file())
            self.assertTrue((results_dir / "query_prediction_02.png").is_file())


if __name__ == "__main__":
    unittest.main()
