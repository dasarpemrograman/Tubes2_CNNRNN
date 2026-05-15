from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import f1_score

try:
    from cnn.configs import (
        DEFAULT_BATCH_SIZE,
        INTEL_CLASS_NAMES,
        generate_required_cnn_configs,
    )
    from cnn.layers import scratch_model_from_keras_model
    from cnn.preprocessing import load_image_batch
    from cnn.training import (
        TrainingRunConfig,
        aggregate_shared_conv2d_results,
        config_with_model_id,
        create_image_sequence,
        limit_examples,
        read_dataset_index,
        read_json,
        split_dataset_examples,
        train_one_cnn_config,
        write_json,
        write_ranked_results,
        write_results_csv,
    )
except ModuleNotFoundError:
    from configs import DEFAULT_BATCH_SIZE, INTEL_CLASS_NAMES, generate_required_cnn_configs
    from layers import scratch_model_from_keras_model
    from preprocessing import load_image_batch
    from training import (
        TrainingRunConfig,
        aggregate_shared_conv2d_results,
        config_with_model_id,
        create_image_sequence,
        limit_examples,
        read_dataset_index,
        read_json,
        split_dataset_examples,
        train_one_cnn_config,
        write_json,
        write_ranked_results,
        write_results_csv,
    )


def aggregate_cnn_results(artifacts_dir: Path) -> list[dict[str, object]]:
    results = aggregate_shared_conv2d_results(artifacts_dir)
    write_results_csv(results, artifacts_dir / "metrics" / "results.csv")
    write_ranked_results(results, artifacts_dir)
    return results


def evaluate_keras_vs_scratch(
    artifacts_dir: Path,
    dataset_index: Path,
    model_id: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_test_samples: int | None = None,
    seed: int = 42,
) -> dict[str, object]:
    import keras

    # Importing cnn.models registers the custom LocallyConnected2D class for Keras load_model.
    import cnn.models  # noqa: F401

    best = get_best_shared_result(artifacts_dir, model_id)
    selected_model_id = str(best["model_id"])
    image_size = (int(best["image_height"]), int(best["image_width"]))
    model_path = artifacts_dir / "models" / f"{selected_model_id}.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained Keras model not found: {model_path}")

    examples = read_dataset_index(dataset_index)
    _, _, test_examples = split_dataset_examples(examples, validation_fraction=0.2, seed=seed)
    test_examples = limit_examples(test_examples, max_test_samples, seed)

    keras_model = keras.models.load_model(model_path)
    keras_sequence = create_image_sequence(
        test_examples,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    keras_probabilities = keras_model.predict(keras_sequence, verbose=0)
    scratch_probabilities = predict_scratch_batches(
        keras_model,
        test_examples,
        image_size=image_size,
        batch_size=batch_size,
    )

    output_rows, mismatches, metrics = compare_prediction_arrays(
        test_examples,
        keras_probabilities,
        scratch_probabilities,
        selected_model_id,
    )
    prediction_path = artifacts_dir / "predictions" / f"{selected_model_id}_keras_vs_scratch.csv"
    mismatch_path = artifacts_dir / "predictions" / f"{selected_model_id}_keras_scratch_mismatches.csv"
    metric_path = artifacts_dir / "metrics" / f"{selected_model_id}_keras_vs_scratch.json"
    write_dict_rows(prediction_path, output_rows)
    write_dict_rows(mismatch_path, mismatches)
    write_json(metric_path, metrics)
    return metrics


def run_shared_vs_non_shared(
    artifacts_dir: Path,
    dataset_index: Path,
    model_id: str | None = None,
    image_size: tuple[int, int] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = 10,
    validation_fraction: float = 0.2,
    seed: int = 42,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    max_test_samples: int | None = None,
    force: bool = False,
) -> dict[str, object]:
    shared_best = get_best_shared_result(artifacts_dir, model_id)
    shared_model_id = str(shared_best["model_id"])
    shared_config = get_required_config(shared_model_id)
    non_shared_id = f"{shared_model_id}_non_shared"
    non_shared_config = config_with_model_id(shared_config, non_shared_id)

    effective_image_size = image_size or (int(shared_best["image_height"]), int(shared_best["image_width"]))
    examples = read_dataset_index(dataset_index)
    train_examples, validation_examples, test_examples = split_dataset_examples(
        examples,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    train_examples = limit_examples(train_examples, max_train_samples, seed)
    validation_examples = limit_examples(validation_examples, max_validation_samples, seed)
    test_examples = limit_examples(test_examples, max_test_samples, seed)

    run_config = TrainingRunConfig(
        dataset_index=dataset_index,
        artifacts_dir=artifacts_dir,
        image_size=effective_image_size,
        batch_size=batch_size,
        epochs=epochs,
        validation_fraction=validation_fraction,
        seed=seed,
        max_train_samples=max_train_samples,
        max_validation_samples=max_validation_samples,
        max_test_samples=max_test_samples,
        force=force,
        shared_parameters=False,
    )
    non_shared_result = train_one_cnn_config(
        non_shared_config,
        run_config,
        train_examples,
        validation_examples,
        test_examples,
    )

    comparison = {
        "shared_model_id": shared_model_id,
        "non_shared_model_id": non_shared_id,
        "shared_test_macro_f1": float(shared_best["test_macro_f1"]),
        "non_shared_test_macro_f1": float(non_shared_result["test_macro_f1"]),
        "shared_parameter_count": int(shared_best["parameter_count"]),
        "non_shared_parameter_count": int(non_shared_result["parameter_count"]),
        "shared_validation_macro_f1": float(shared_best["validation_macro_f1"]),
        "non_shared_validation_macro_f1": float(non_shared_result["validation_macro_f1"]),
    }
    write_json(artifacts_dir / "metrics" / "shared_vs_non_shared.json", comparison)
    write_results_csv([comparison], artifacts_dir / "metrics" / "shared_vs_non_shared.csv")
    return comparison


def get_best_shared_result(artifacts_dir: Path, model_id: str | None = None) -> dict[str, Any]:
    if model_id is not None:
        metric_path = artifacts_dir / "metrics" / f"{model_id}.json"
        if not metric_path.is_file():
            raise FileNotFoundError(f"Metric file not found for model {model_id}: {metric_path}")
        return read_json(metric_path)

    best_path = artifacts_dir / "metrics" / "best_shared_conv2d.json"
    if best_path.is_file():
        return read_json(best_path)

    results = aggregate_cnn_results(artifacts_dir)
    if not results:
        raise FileNotFoundError("No shared Conv2D metrics found. Train at least one config first.")
    return max(results, key=lambda row: float(row["test_macro_f1"]))


def get_required_config(model_id: str):
    for config in generate_required_cnn_configs():
        if config.model_id == model_id:
            return config
    raise ValueError(f"Model id is not one of the required shared Conv2D configs: {model_id}")


def predict_scratch_batches(
    keras_model: Any,
    examples: Sequence[Any],
    image_size: tuple[int, int],
    batch_size: int,
) -> np.ndarray:
    scratch_model = scratch_model_from_keras_model(keras_model)
    outputs: list[np.ndarray] = []
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        images = load_image_batch([example.filepath for example in batch_examples], image_size)
        outputs.append(scratch_model.predict(images))
    return np.concatenate(outputs, axis=0)


def compare_prediction_arrays(
    examples: Sequence[Any],
    keras_probabilities: np.ndarray,
    scratch_probabilities: np.ndarray,
    model_id: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    true_ids = np.asarray([example.label_id for example in examples], dtype=np.int64)
    keras_ids = keras_probabilities.argmax(axis=1)
    scratch_ids = scratch_probabilities.argmax(axis=1)
    max_probability_diff = float(np.max(np.abs(keras_probabilities - scratch_probabilities)))

    rows: list[dict[str, object]] = []
    mismatches: list[dict[str, object]] = []
    for example, keras_id, scratch_id, keras_probs, scratch_probs in zip(
        examples,
        keras_ids,
        scratch_ids,
        keras_probabilities,
        scratch_probabilities,
    ):
        row = {
            "filepath": str(example.filepath),
            "split": example.split,
            "label_name": example.label_name,
            "label_id": example.label_id,
            "keras_predicted_label_id": int(keras_id),
            "keras_predicted_label_name": INTEL_CLASS_NAMES[int(keras_id)],
            "keras_confidence": float(keras_probs[int(keras_id)]),
            "scratch_predicted_label_id": int(scratch_id),
            "scratch_predicted_label_name": INTEL_CLASS_NAMES[int(scratch_id)],
            "scratch_confidence": float(scratch_probs[int(scratch_id)]),
            "predictions_match": int(keras_id == scratch_id),
        }
        rows.append(row)
        if keras_id != scratch_id:
            mismatches.append(row)

    metrics = {
        "model_id": model_id,
        "keras_macro_f1": float(f1_score(true_ids, keras_ids, average="macro")),
        "scratch_macro_f1": float(f1_score(true_ids, scratch_ids, average="macro")),
        "prediction_matches": int(np.sum(keras_ids == scratch_ids)),
        "prediction_mismatches": int(np.sum(keras_ids != scratch_ids)),
        "total_examples": len(examples),
        "max_probability_abs_diff": max_probability_diff,
    }
    return rows, mismatches, metrics


def write_dict_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = [
            "filepath",
            "split",
            "label_name",
            "label_id",
            "keras_predicted_label_id",
            "keras_predicted_label_name",
            "keras_confidence",
            "scratch_predicted_label_id",
            "scratch_predicted_label_name",
            "scratch_confidence",
            "predictions_match",
        ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    parser = argparse.ArgumentParser(description="CNN result aggregation and evaluation commands.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--artifacts-dir", type=Path, default=root / "artifacts" / "cnn")

    scratch_parser = subparsers.add_parser("keras-vs-scratch")
    scratch_parser.add_argument("--artifacts-dir", type=Path, default=root / "artifacts" / "cnn")
    scratch_parser.add_argument("--dataset-index", type=Path, default=root / "artifacts" / "cnn" / "dataset_index.csv")
    scratch_parser.add_argument("--model-id")
    scratch_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    scratch_parser.add_argument("--max-test-samples", type=int)
    scratch_parser.add_argument("--seed", type=int, default=42)

    non_shared_parser = subparsers.add_parser("shared-vs-non-shared")
    non_shared_parser.add_argument("--artifacts-dir", type=Path, default=root / "artifacts" / "cnn")
    non_shared_parser.add_argument("--dataset-index", type=Path, default=root / "artifacts" / "cnn" / "dataset_index.csv")
    non_shared_parser.add_argument("--model-id")
    non_shared_parser.add_argument("--image-size", type=_parse_size)
    non_shared_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    non_shared_parser.add_argument("--epochs", type=int, default=10)
    non_shared_parser.add_argument("--validation-fraction", type=float, default=0.2)
    non_shared_parser.add_argument("--seed", type=int, default=42)
    non_shared_parser.add_argument("--max-train-samples", type=int)
    non_shared_parser.add_argument("--max-validation-samples", type=int)
    non_shared_parser.add_argument("--max-test-samples", type=int)
    non_shared_parser.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "aggregate":
        results = aggregate_cnn_results(args.artifacts_dir)
        print(f"Aggregated {len(results)} shared Conv2D result(s).")
    elif args.command == "keras-vs-scratch":
        metrics = evaluate_keras_vs_scratch(
            artifacts_dir=args.artifacts_dir,
            dataset_index=args.dataset_index,
            model_id=args.model_id,
            batch_size=args.batch_size,
            max_test_samples=args.max_test_samples,
            seed=args.seed,
        )
        print(f"Keras vs scratch metrics: {metrics}")
    elif args.command == "shared-vs-non-shared":
        comparison = run_shared_vs_non_shared(
            artifacts_dir=args.artifacts_dir,
            dataset_index=args.dataset_index,
            model_id=args.model_id,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            max_train_samples=args.max_train_samples,
            max_validation_samples=args.max_validation_samples,
            max_test_samples=args.max_test_samples,
            force=args.force,
        )
        print(f"Shared vs non-shared comparison: {comparison}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
