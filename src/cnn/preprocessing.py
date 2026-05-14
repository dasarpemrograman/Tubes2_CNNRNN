from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

try:
    from cnn.configs import (
        CNNPreprocessingConfig,
        DEFAULT_BATCH_SIZE,
        DEFAULT_IMAGE_SIZE,
        INTEL_CLASS_NAMES,
    )
except ModuleNotFoundError:
    from configs import (
        CNNPreprocessingConfig,
        DEFAULT_BATCH_SIZE,
        DEFAULT_IMAGE_SIZE,
        INTEL_CLASS_NAMES,
    )

DATASET_INDEX_FIELDS = ["filepath", "split", "label_name", "label_id"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class IntelDatasetRow:
    filepath: str
    split: str
    label_name: str
    label_id: int

    def as_dict(self) -> dict[str, str | int]:
        return {
            "filepath": self.filepath,
            "split": self.split,
            "label_name": self.label_name,
            "label_id": self.label_id,
        }


def build_label_mapping(class_names: Sequence[str] = INTEL_CLASS_NAMES) -> dict[str, int]:
    return {label_name: label_id for label_id, label_name in enumerate(class_names)}


def load_image(
    image_path: str | Path,
    target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Load one image as RGB float32 array normalized to [0, 1]."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    with Image.open(path) as image:
        image = image.convert("RGB")
        height, width = target_size
        image = image.resize((width, height))
        array = np.asarray(image, dtype=np.float32) / 255.0

    return array


def load_image_batch(
    image_paths: Sequence[str | Path],
    target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Load image paths into a batch with shape (N, H, W, 3)."""
    images = [load_image(path, target_size=target_size) for path in image_paths]
    if not images:
        height, width = target_size
        return np.empty((0, height, width, 3), dtype=np.float32)
    return np.stack(images, axis=0).astype(np.float32, copy=False)


def iter_image_batches(
    image_paths: Sequence[str | Path],
    target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterable[np.ndarray]:
    """Yield image arrays in batches of at most batch_size."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, len(image_paths), batch_size):
        yield load_image_batch(image_paths[start : start + batch_size], target_size=target_size)


def build_intel_dataset_index(
    dataset_root: str | Path,
    class_names: Sequence[str] = INTEL_CLASS_NAMES,
) -> list[IntelDatasetRow]:
    """Build an index for Intel Image Classification folders.

    Expected split folders are commonly named ``seg_train``, ``seg_test``, and
    optionally ``seg_pred``. Only labeled class subfolders are indexed, so
    ``seg_pred`` is skipped unless it contains the expected class directories.
    """
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    label_to_id = build_label_mapping(class_names)
    rows: list[IntelDatasetRow] = []

    for split_dir in _iter_split_dirs(root):
        split_name = split_dir.name
        for label_name, label_id in label_to_id.items():
            label_dir = split_dir / label_name
            if not label_dir.is_dir():
                continue
            for image_path in _iter_image_files(label_dir):
                rows.append(
                    IntelDatasetRow(
                        filepath=_index_filepath(image_path),
                        split=split_name,
                        label_name=label_name,
                        label_id=label_id,
                    )
                )

    if not rows:
        expected = ", ".join(class_names)
        raise ValueError(
            f"No labeled Intel images found under {root}. "
            f"Expected split folders containing class folders: {expected}."
        )

    return rows


def write_dataset_index(rows: Iterable[IntelDatasetRow], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASET_INDEX_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def write_label_mapping(mapping: dict[str, int], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")


def build_and_save_intel_index(
    config: CNNPreprocessingConfig,
    index_output: str | Path,
    labels_output: str | Path,
) -> list[IntelDatasetRow]:
    rows = build_intel_dataset_index(config.dataset_root)
    write_dataset_index(rows, index_output)
    write_label_mapping(build_label_mapping(), labels_output)
    return rows


def _iter_split_dirs(dataset_root: Path) -> list[Path]:
    preferred_names = ("seg_train", "seg_test", "seg_pred", "train", "val", "valid", "test")
    preferred = [dataset_root / name for name in preferred_names if (dataset_root / name).is_dir()]
    if preferred:
        return preferred
    return [path for path in sorted(dataset_root.iterdir()) if path.is_dir()]


def _iter_image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _index_filepath(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_dataset_root(project_root: Path) -> Path:
    candidates = (
        project_root / "data" / "intel_Image_classification",
        project_root / "data" / "intel_image_classification",
        project_root / "datasets" / "intel_Image_classification",
        project_root / "datasets" / "intel_image_classification",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _parse_size(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Image size must use HxW format, for example 150x150.")
    height, width = (int(part) for part in parts)
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("Image height and width must be positive.")
    return height, width


def main(argv: Sequence[str] | None = None) -> int:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Build Intel Image Classification dataset index.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_default_dataset_root(root),
        help="Path to the Intel Image Classification dataset root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "artifacts" / "cnn" / "dataset_index.csv",
        help="CSV output path for the generated index.",
    )
    parser.add_argument(
        "--labels-output",
        type=Path,
        default=root / "artifacts" / "cnn" / "labels.json",
        help="JSON output path for the class-to-id label mapping.",
    )
    parser.add_argument(
        "--image-size",
        type=_parse_size,
        default=DEFAULT_IMAGE_SIZE,
        help="Optional loader smoke-test size in HxW format. Default: 150x150.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Configured batch size for downstream loaders and smoke checks.",
    )
    parser.add_argument(
        "--smoke-load",
        action="store_true",
        help="Load the first indexed image to verify PIL/NumPy preprocessing.",
    )
    args = parser.parse_args(argv)

    config = CNNPreprocessingConfig(
        dataset_root=args.dataset_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )
    rows = build_and_save_intel_index(config, args.output, args.labels_output)

    if args.smoke_load:
        batch = next(
            iter_image_batches(
                [row.filepath for row in rows],
                target_size=config.image_size,
                batch_size=config.batch_size,
            )
        )
        print(
            f"Loaded batch shape={batch.shape} dtype={batch.dtype} "
            f"range=({batch.min()}, {batch.max()})"
        )

    print(f"Wrote {len(rows)} indexed images to {args.output}")
    print(f"Wrote labels to {args.labels_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
