from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cnn.configs import DEFAULT_BATCH_SIZE, DEFAULT_IMAGE_SIZE, INTEL_CLASS_NAMES  # noqa: E402
from cnn.layers import scratch_model_from_keras_model  # noqa: E402
from cnn.preprocessing import load_image_batch  # noqa: E402
from cnn.training import DatasetExample, read_dataset_index  # noqa: E402


def run_scratch_inference(
    keras_model_path: Path,
    dataset_index: Path,
    output_path: Path,
    image_size: tuple[int, int],
    batch_size: int,
    limit: int | None,
) -> None:
    import keras

    keras_model = keras.models.load_model(keras_model_path)
    scratch_model = scratch_model_from_keras_model(keras_model)

    examples = read_dataset_index(dataset_index)
    if limit is not None:
        examples = examples[:limit]
    if not examples:
        raise ValueError("No examples available for scratch inference.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filepath",
                "split",
                "label_name",
                "label_id",
                "predicted_label_id",
                "predicted_label_name",
                "confidence",
            ],
        )
        writer.writeheader()
        for batch_examples in _batched(examples, batch_size):
            images = load_image_batch([example.filepath for example in batch_examples], image_size)
            probabilities = scratch_model.predict(images)
            predicted_ids = probabilities.argmax(axis=1)
            for example, predicted_id, probability in zip(batch_examples, predicted_ids, probabilities):
                writer.writerow(
                    {
                        "filepath": str(example.filepath),
                        "split": example.split,
                        "label_name": example.label_name,
                        "label_id": example.label_id,
                        "predicted_label_id": int(predicted_id),
                        "predicted_label_name": INTEL_CLASS_NAMES[int(predicted_id)],
                        "confidence": float(probability[int(predicted_id)]),
                    }
                )


def _batched(examples: Sequence[DatasetExample], batch_size: int) -> list[Sequence[DatasetExample]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return [examples[start : start + batch_size] for start in range(0, len(examples), batch_size)]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
    parser = argparse.ArgumentParser(description="Run NumPy scratch CNN inference from a Keras model.")
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "artifacts" / "cnn" / "models" / "cnn_c2_f32-64_k3-3_max.keras",
        help="Path to a trained .keras model.",
    )
    parser.add_argument(
        "--dataset-index",
        type=Path,
        default=root / "artifacts" / "cnn" / "dataset_index.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "artifacts" / "cnn" / "predictions" / "scratch_predictions.csv",
    )
    parser.add_argument("--image-size", type=_parse_size, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args(argv)

    if not args.model.is_file():
        raise FileNotFoundError(
            f"Keras model not found: {args.model}. Run CNN training first or pass --model."
        )
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided.")

    run_scratch_inference(
        keras_model_path=args.model,
        dataset_index=args.dataset_index,
        output_path=args.output,
        image_size=args.image_size,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    print(f"Wrote scratch predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
