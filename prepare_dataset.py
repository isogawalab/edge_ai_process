from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class DatasetRow:
    split: str
    relative_path: str
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a class-directory image dataset into train/valid directories "
            "and save matching CSV files."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input root directory. Expected layout: input/<class>/*.{png,jpg,...}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output root directory for id_train/, id_valid/, train.csv, valid.csv.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.2,
        help="Fraction of images per class assigned to validation. Default: 0.2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used before splitting. Default: 42",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=10,
        help="Minimum required image count per class. Default: 10",
    )
    parser.add_argument(
        "--recommended-images",
        type=int,
        default=20,
        help="Recommended image count per class. Default: 20",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite id_train/, id_valid/, train.csv, and valid.csv in the output.",
    )
    return parser.parse_args()


def ensure_valid_ratio(valid_ratio: float) -> None:
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError("--valid-ratio must be greater than 0 and less than 1.")


def list_class_directories(input_root: Path) -> list[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_root}")

    class_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class directories found under: {input_root}")
    return class_dirs


def list_images(class_dir: Path) -> list[Path]:
    images = sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No supported image files found in class directory: {class_dir}")
    return images


def validate_output_root(output_root: Path, force: bool) -> None:
    managed_paths = [
        output_root / "id_train",
        output_root / "id_valid",
        output_root / "train.csv",
        output_root / "valid.csv",
    ]
    existing_paths = [path for path in managed_paths if path.exists()]

    if existing_paths and not force:
        joined = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            "Output already contains generated dataset artifacts. "
            f"Use --force to replace them: {joined}"
        )

    if force:
        for path in managed_paths:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

    output_root.mkdir(parents=True, exist_ok=True)


def split_count(total_count: int, valid_ratio: float) -> int:
    valid_count = int(round(total_count * valid_ratio))
    return max(1, min(total_count - 1, valid_count))


def copy_split(
    images: list[Path],
    label: str,
    split: str,
    destination_root: Path,
) -> list[DatasetRow]:
    destination_dir = destination_root / split / label
    destination_dir.mkdir(parents=True, exist_ok=True)

    rows: list[DatasetRow] = []
    for image_path in images:
        destination_path = destination_dir / image_path.name
        shutil.copy2(image_path, destination_path)
        rows.append(
            DatasetRow(
                split=split,
                relative_path=destination_path.relative_to(destination_root).as_posix(),
                label=label,
            )
        )
    return rows


def write_csv(csv_path: Path, rows: list[DatasetRow]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["x", "y"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"x": row.relative_path, "y": row.label})


def build_dataset(args: argparse.Namespace) -> None:
    input_root = args.input.resolve()
    output_root = args.output.resolve()

    ensure_valid_ratio(args.valid_ratio)
    validate_output_root(output_root, args.force)

    class_dirs = list_class_directories(input_root)
    randomizer = random.Random(args.seed)

    train_rows: list[DatasetRow] = []
    valid_rows: list[DatasetRow] = []
    warnings: list[str] = []

    for class_dir in class_dirs:
        label = class_dir.name
        images = list_images(class_dir)
        image_count = len(images)

        if image_count < args.min_images:
            raise ValueError(
                f"Class '{label}' has {image_count} images, which is below the minimum "
                f"required count of {args.min_images}."
            )
        if image_count < args.recommended_images:
            warnings.append(
                f"Warning: class '{label}' has {image_count} images. "
                f"{args.recommended_images} or more is recommended."
            )

        shuffled_images = images[:]
        randomizer.shuffle(shuffled_images)

        valid_count = split_count(image_count, args.valid_ratio)
        valid_images = shuffled_images[:valid_count]
        train_images = shuffled_images[valid_count:]

        train_rows.extend(copy_split(train_images, label, "id_train", output_root))
        valid_rows.extend(copy_split(valid_images, label, "id_valid", output_root))

        print(
            f"class={label} total={image_count} "
            f"train={len(train_images)} valid={len(valid_images)}"
        )

    train_rows.sort(key=lambda row: row.relative_path)
    valid_rows.sort(key=lambda row: row.relative_path)

    write_csv(output_root / "train.csv", train_rows)
    write_csv(output_root / "valid.csv", valid_rows)

    print(f"Saved train.csv with {len(train_rows)} rows: {output_root / 'train.csv'}")
    print(f"Saved valid.csv with {len(valid_rows)} rows: {output_root / 'valid.csv'}")

    for warning in warnings:
        print(warning)


def main() -> None:
    args = parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
