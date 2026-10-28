"""Cấu hình mặc định cho demo EMNIST ByClass."""

from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
DATA_DIR = DEMO_DIR / "data" / "EMNIST"
RESULTS_DIR = DEMO_DIR / "results"

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
NUM_CLASSES = 62
DEFAULT_CLASS_IDS = tuple(range(NUM_CLASSES))
MAX_TRAIN_PER_CLASS = 200
WEB_MAX_TRAIN_PER_CLASS = 500
MAX_TEST_PER_CLASS = 20
VALIDATION_FRACTION = 0.2
MAX_VALIDATION_PER_CLASS = 10
SAMPLE_IMAGE_COUNT = 8
RANDOM_STATE = 42
K_CANDIDATES = (1, 3, 5, 7, 9)
FIXED_EXPERIMENT_K = 101

IDX_FILENAMES = {
    "train_images": "emnist-byclass-train-images-idx3-ubyte.gz",
    "train_labels": "emnist-byclass-train-labels-idx1-ubyte.gz",
    "test_images": "emnist-byclass-test-images-idx3-ubyte.gz",
    "test_labels": "emnist-byclass-test-labels-idx1-ubyte.gz",
    "mapping": "emnist-byclass-mapping.txt",
}
