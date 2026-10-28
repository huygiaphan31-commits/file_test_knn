import tempfile
import unittest
from pathlib import Path

from data_loader import load_idx_subset, load_character_mapping
from tests.helpers import write_complete_mapping, write_idx_pair


class DataLoaderTests(unittest.TestCase):
    def test_streams_idx_and_respects_per_class_cap(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            images_path, labels_path = write_idx_pair(
                folder, "train", {0: 5, 1: 4}
            )
            dataset = load_idx_subset(
                images_path, labels_path, (0, 1), max_per_class=2, random_state=12
            )
            self.assertEqual(dataset.images.shape, (4, 28, 28))
            self.assertEqual(dataset.observed_class_counts, {0: 5, 1: 4})
            self.assertEqual(dataset.labels.tolist().count(0), 2)
            self.assertEqual(dataset.labels.tolist().count(1), 2)

    def test_reads_complete_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mapping.txt"
            write_complete_mapping(path)
            mapping = load_character_mapping(path)
            self.assertEqual(len(mapping), 62)
            self.assertEqual(mapping[0], "0")
            self.assertEqual(mapping[10], "A")
            self.assertEqual(mapping[36], "a")

    def test_mapping_decodes_digit_and_letter_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mapping.txt"
            write_complete_mapping(path)

            mapping = load_character_mapping(path)

        self.assertEqual(
            {class_id: mapping[class_id] for class_id in (0, 9, 10, 35, 36, 61)},
            {0: "0", 9: "9", 10: "A", 35: "Z", 36: "a", 61: "z"},
        )
        self.assertEqual(
            set(mapping.values()),
            set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"),
        )

    def test_accepts_valid_character_set_without_assuming_class_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mapping.txt"
            write_complete_mapping(path)
            lines = path.read_text(encoding="ascii").splitlines()
            lines[0] = "0 65"
            lines[10] = "10 48"
            path.write_text("\n".join(lines) + "\n", encoding="ascii")

            mapping = load_character_mapping(path)

            expected_characters = set(
                "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            )
            self.assertEqual(set(mapping.values()), expected_characters)
            self.assertEqual(mapping[0], "A")
            self.assertEqual(mapping[10], "0")

    def test_rejects_duplicate_character(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mapping.txt"
            write_complete_mapping(path)
            lines = path.read_text(encoding="ascii").splitlines()
            lines[1] = "1 48"
            path.write_text("\n".join(lines) + "\n", encoding="ascii")

            with self.assertRaisesRegex(ValueError, "ký tự bị trùng"):
                load_character_mapping(path)

    def test_rejects_character_outside_allowed_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mapping.txt"
            write_complete_mapping(path)
            lines = path.read_text(encoding="ascii").splitlines()
            lines[0] = "0 33"
            path.write_text("\n".join(lines) + "\n", encoding="ascii")

            with self.assertRaisesRegex(ValueError, "đúng các ký tự"):
                load_character_mapping(path)

    def test_rejects_missing_class_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mapping.txt"
            write_complete_mapping(path)
            lines = path.read_text(encoding="ascii").splitlines()
            path.write_text("\n".join(lines[:-1]) + "\n", encoding="ascii")

            with self.assertRaisesRegex(ValueError, "đúng 62 class ID"):
                load_character_mapping(path)

    def test_rejects_duplicate_class_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mapping.txt"
            write_complete_mapping(path)
            lines = path.read_text(encoding="ascii").splitlines()
            lines[1] = "0 49"
            path.write_text("\n".join(lines) + "\n", encoding="ascii")

            with self.assertRaisesRegex(ValueError, "class_id 0 bị lặp"):
                load_character_mapping(path)

    def test_rejects_empty_class(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            images_path, labels_path = write_idx_pair(folder, "train", {0: 2})
            with self.assertRaisesRegex(ValueError, "Không có mẫu"):
                load_idx_subset(images_path, labels_path, (0, 1), 2)


if __name__ == "__main__":
    unittest.main()
