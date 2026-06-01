from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "prepare_dataset.py"
REFERENCE_ROOT = PROJECT_ROOT / "numbers_dataset"
REFERENCE_TRAIN_ROOT = REFERENCE_ROOT / "id_train"
REFERENCE_VALID_ROOT = REFERENCE_ROOT / "id_valid"


def iter_class_names() -> list[str]:
    names: set[str] = set()
    for split_root in (REFERENCE_TRAIN_ROOT, REFERENCE_VALID_ROOT):
        for path in split_root.iterdir():
            if path.is_dir() and not path.name.endswith(".cache"):
                names.add(path.name)
    return sorted(names)


def count_images(directory: Path) -> int:
    return sum(1 for path in directory.rglob("*") if path.is_file())


class PrepareDatasetCliTest(unittest.TestCase):
    def test_builds_split_dataset_from_reference_dataset_copy(self) -> None:
        class_names = iter_class_names()
        self.assertTrue(class_names, "No class directories found in reference dataset.")

        with tempfile.TemporaryDirectory() as temp_source_dir, tempfile.TemporaryDirectory() as temp_output_dir:
            source_root = Path(temp_source_dir)
            output_root = Path(temp_output_dir)

            expected_totals: dict[str, int] = {}

            for class_name in class_names:
                merged_class_dir = source_root / class_name
                merged_class_dir.mkdir(parents=True, exist_ok=True)

                copied_count = 0
                for split_root in (REFERENCE_TRAIN_ROOT, REFERENCE_VALID_ROOT):
                    reference_class_dir = split_root / class_name
                    if not reference_class_dir.exists():
                        continue

                    for image_path in sorted(reference_class_dir.iterdir()):
                        if not image_path.is_file():
                            continue

                        destination_path = merged_class_dir / image_path.name
                        self.assertFalse(
                            destination_path.exists(),
                            f"Duplicate file name detected while merging class '{class_name}': {image_path.name}",
                        )
                        shutil.copy2(image_path, destination_path)
                        copied_count += 1

                expected_totals[class_name] = copied_count
                self.assertGreaterEqual(
                    copied_count,
                    10,
                    f"Reference class '{class_name}' does not meet the minimum image requirement.",
                )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--input",
                    str(source_root),
                    "--output",
                    str(output_root),
                    "--valid-ratio",
                    "0.2",
                    "--seed",
                    "42",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("Saved train.csv", completed.stdout)
            self.assertIn("Saved valid.csv", completed.stdout)

            train_root = output_root / "id_train"
            valid_root = output_root / "id_valid"
            train_csv = output_root / "train.csv"
            valid_csv = output_root / "valid.csv"

            self.assertTrue(train_root.is_dir())
            self.assertTrue(valid_root.is_dir())
            self.assertTrue(train_csv.is_file())
            self.assertTrue(valid_csv.is_file())

            with train_csv.open(encoding="utf-8") as train_file:
                train_rows = list(csv.DictReader(train_file))
            with valid_csv.open(encoding="utf-8") as valid_file:
                valid_rows = list(csv.DictReader(valid_file))

            self.assertEqual(count_images(train_root), len(train_rows))
            self.assertEqual(count_images(valid_root), len(valid_rows))

            output_totals = {class_name: 0 for class_name in class_names}
            for rows, expected_prefix in ((train_rows, "id_train/"), (valid_rows, "id_valid/")):
                for row in rows:
                    relative_path = row["x"]
                    label = row["y"]
                    self.assertIn(label, output_totals)
                    self.assertTrue(relative_path.startswith(expected_prefix))

                    image_path = output_root / relative_path
                    self.assertTrue(image_path.is_file(), f"Missing generated image: {relative_path}")
                    self.assertEqual(label, image_path.parent.name)

                    output_totals[label] += 1

            self.assertEqual(expected_totals, output_totals)


if __name__ == "__main__":
    unittest.main()
