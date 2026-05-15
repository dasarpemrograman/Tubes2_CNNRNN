from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

try:
    from cnn.configs import (
        CNNModelConfig,
        DEFAULT_BATCH_SIZE,
        DEFAULT_IMAGE_SIZE,
        INTEL_CLASS_NAMES,
        generate_required_cnn_configs,
    )
    from cnn.preprocessing import load_image_batch
except ModuleNotFoundError:
    from configs import (
        CNNModelConfig,
        DEFAULT_BATCH_SIZE,
        DEFAULT_IMAGE_SIZE,
        INTEL_CLASS_NAMES,
        generate_required_cnn_configs,
    )
    from preprocessing import load_image_batch


@dataclass(frozen=True)
class DatasetExample:
    filepath: Path
    split: str
    label_name: str
    label_id: int


@dataclass(frozen=True)
class TrainingRunConfig:
    dataset_index: Path
    artifacts_dir: Path
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = 10
    validation_fraction: float = 0.2
    seed: int = 42
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    max_test_samples: int | None = None
    force: bool = False
    shared_parameters: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_index", Path(self.dataset_index))
        object.__setattr__(self, "artifacts_dir", Path(self.artifacts_dir))
        if self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise ValueError("image_size must contain positive height and width.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1.")


def train_required_cnn_configs(
    run_config: TrainingRunConfig,
    model_configs: Sequence[CNNModelConfig] | None = None,
) -> list[dict[str, object]]:
    examples = read_dataset_index(run_config.dataset_index)
    train_examples, validation_examples, test_examples = split_dataset_examples(
        examples,
        validation_fraction=run_config.validation_fraction,
        seed=run_config.seed,
    )
    train_examples = limit_examples(train_examples, run_config.max_train_samples, run_config.seed)
    validation_examples = limit_examples(
        validation_examples,
        run_config.max_validation_samples,
        run_config.seed,
    )
    test_examples = limit_examples(test_examples, run_config.max_test_samples, run_config.seed)

    if not train_examples or not validation_examples or not test_examples:
        raise ValueError("Training requires non-empty train, validation, and test splits.")

    configs = list(model_configs or generate_required_cnn_configs())
    ensure_cnn_artifact_dirs(run_config.artifacts_dir)

    results: list[dict[str, object]] = []
    for model_config in configs:
        if is_completed_run(
            model_config,
            run_config,
            train_samples=len(train_examples),
            validation_samples=len(validation_examples),
            test_samples=len(test_examples),
        ):
            print(f"Skipping completed config: {model_config.model_id}")
            results.append(read_json(run_config.artifacts_dir / "metrics" / f"{model_config.model_id}.json"))
            continue
        result = train_one_cnn_config(
            model_config,
            run_config,
            train_examples,
            validation_examples,
            test_examples,
        )
        results.append(result)

    aggregated = aggregate_shared_conv2d_results(run_config.artifacts_dir)
    write_results_csv(aggregated, run_config.artifacts_dir / "metrics" / "results.csv")
    write_ranked_results(aggregated, run_config.artifacts_dir)
    return results


def train_one_cnn_config(
    model_config: CNNModelConfig,
    run_config: TrainingRunConfig,
    train_examples: Sequence[DatasetExample],
    validation_examples: Sequence[DatasetExample],
    test_examples: Sequence[DatasetExample],
) -> dict[str, object]:
    import keras

    try:
        from cnn.models import build_keras_cnn_model
    except ModuleNotFoundError:
        from models import build_keras_cnn_model

    keras.utils.set_random_seed(run_config.seed)

    train_sequence = create_image_sequence(
        train_examples,
        image_size=run_config.image_size,
        batch_size=run_config.batch_size,
        shuffle=True,
        seed=run_config.seed,
    )
    validation_sequence = create_image_sequence(
        validation_examples,
        image_size=run_config.image_size,
        batch_size=run_config.batch_size,
        shuffle=False,
        seed=run_config.seed,
    )
    test_sequence = create_image_sequence(
        test_examples,
        image_size=run_config.image_size,
        batch_size=run_config.batch_size,
        shuffle=False,
        seed=run_config.seed,
    )

    model = build_keras_cnn_model(
        model_config,
        input_shape=(*run_config.image_size, 3),
        num_classes=len(INTEL_CLASS_NAMES),
        shared_parameters=run_config.shared_parameters,
    )

    config_dir = run_config.artifacts_dir / "configs"
    model_dir = run_config.artifacts_dir / "models"
    history_dir = run_config.artifacts_dir / "histories"
    prediction_dir = run_config.artifacts_dir / "predictions"
    metric_dir = run_config.artifacts_dir / "metrics"
    plot_dir = run_config.artifacts_dir / "plots"

    write_json(config_dir / f"{model_config.model_id}.json", model_config.as_dict())
    write_model_summary(model, metric_dir / f"{model_config.model_id}_summary.txt")

    history = model.fit(
        train_sequence,
        validation_data=validation_sequence,
        epochs=run_config.epochs,
        verbose=2,
    )

    model.save(model_dir / f"{model_config.model_id}.keras")
    write_json(history_dir / f"{model_config.model_id}.json", history.history)
    plot_training_history(history.history, plot_dir / f"{model_config.model_id}.png")

    validation_metrics = predict_and_save(
        model,
        validation_sequence,
        validation_examples,
        prediction_dir / f"{model_config.model_id}_validation.csv",
    )
    test_metrics = predict_and_save(
        model,
        test_sequence,
        test_examples,
        prediction_dir / f"{model_config.model_id}_test.csv",
    )

    result = {
        "model_id": model_config.model_id,
        "parameter_sharing": "shared" if run_config.shared_parameters else "non_shared",
        "parameter_count": int(model.count_params()),
        "validation_macro_f1": validation_metrics["macro_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "train_samples": len(train_examples),
        "validation_samples": len(validation_examples),
        "test_samples": len(test_examples),
        "epochs": run_config.epochs,
        "image_height": run_config.image_size[0],
        "image_width": run_config.image_size[1],
        "batch_size": run_config.batch_size,
    }
    write_json(metric_dir / f"{model_config.model_id}.json", result)
    return result


def is_completed_run(
    model_config: CNNModelConfig,
    run_config: TrainingRunConfig,
    train_samples: int,
    validation_samples: int,
    test_samples: int,
) -> bool:
    if run_config.force:
        return False
    model_id = model_config.model_id
    required_paths = [
        run_config.artifacts_dir / "configs" / f"{model_id}.json",
        run_config.artifacts_dir / "models" / f"{model_id}.keras",
        run_config.artifacts_dir / "histories" / f"{model_id}.json",
        run_config.artifacts_dir / "predictions" / f"{model_id}_validation.csv",
        run_config.artifacts_dir / "predictions" / f"{model_id}_test.csv",
        run_config.artifacts_dir / "metrics" / f"{model_id}.json",
        run_config.artifacts_dir / "plots" / f"{model_id}.png",
    ]
    if not all(path.exists() for path in required_paths):
        return False

    metric = read_json(run_config.artifacts_dir / "metrics" / f"{model_id}.json")
    return (
        metric.get("epochs") == run_config.epochs
        and metric.get("image_height") == run_config.image_size[0]
        and metric.get("image_width") == run_config.image_size[1]
        and metric.get("batch_size") == run_config.batch_size
        and metric.get("parameter_sharing", "shared") == (
            "shared" if run_config.shared_parameters else "non_shared"
        )
        and metric.get("train_samples") == train_samples
        and metric.get("validation_samples") == validation_samples
        and metric.get("test_samples") == test_samples
    )


def create_image_sequence(
    examples: Sequence[DatasetExample],
    image_size: tuple[int, int],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Any:
    import keras

    class ImageClassificationSequence(keras.utils.Sequence):
        def __init__(self) -> None:
            super().__init__()
            self.examples = list(examples)
            self.indices = list(range(len(self.examples)))
            self.random = random.Random(seed)
            self.on_epoch_end()

        def __len__(self) -> int:
            return int(np.ceil(len(self.examples) / batch_size))

        def __getitem__(self, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
            start = batch_index * batch_size
            end = start + batch_size
            batch_indices = self.indices[start:end]
            batch_examples = [self.examples[index] for index in batch_indices]
            images = load_image_batch([example.filepath for example in batch_examples], image_size)
            labels = np.asarray([example.label_id for example in batch_examples], dtype=np.int64)
            return images, labels

        def on_epoch_end(self) -> None:
            if shuffle:
                self.random.shuffle(self.indices)

    return ImageClassificationSequence()


def read_dataset_index(index_path: Path) -> list[DatasetExample]:
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Dataset index not found: {index_path}. "
            "Run `uv run python src\\cnn\\preprocessing.py` first."
        )

    project_root = Path.cwd()
    examples: list[DatasetExample] = []
    with index_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_fields = {"filepath", "split", "label_name", "label_id"}
        if not required_fields.issubset(reader.fieldnames or set()):
            raise ValueError(f"Dataset index must contain fields: {sorted(required_fields)}")
        for row in reader:
            filepath = Path(row["filepath"])
            if not filepath.is_absolute():
                filepath = project_root / filepath
            if not filepath.is_file():
                raise FileNotFoundError(f"Indexed image file not found: {filepath}")
            examples.append(
                DatasetExample(
                    filepath=filepath,
                    split=row["split"],
                    label_name=row["label_name"],
                    label_id=int(row["label_id"]),
                )
            )
    if not examples:
        raise ValueError(f"Dataset index is empty: {index_path}")
    return examples


def split_dataset_examples(
    examples: Sequence[DatasetExample],
    validation_fraction: float,
    seed: int,
) -> tuple[list[DatasetExample], list[DatasetExample], list[DatasetExample]]:
    train_examples = [example for example in examples if _is_train_split(example.split)]
    validation_examples = [example for example in examples if _is_validation_split(example.split)]
    test_examples = [example for example in examples if _is_test_split(example.split)]

    if not train_examples:
        raise ValueError("Dataset index has no train split rows.")
    if not test_examples:
        raise ValueError("Dataset index has no test split rows.")

    if not validation_examples:
        labels = [example.label_id for example in train_examples]
        train_examples, validation_examples = train_test_split(
            train_examples,
            test_size=validation_fraction,
            random_state=seed,
            stratify=labels,
        )

    return list(train_examples), list(validation_examples), list(test_examples)


def limit_examples(
    examples: Sequence[DatasetExample],
    max_samples: int | None,
    seed: int,
) -> list[DatasetExample]:
    examples = list(examples)
    if max_samples is None or max_samples >= len(examples):
        return examples
    if max_samples <= 0:
        raise ValueError("sample limits must be positive when provided.")

    labels = [example.label_id for example in examples]
    unique_label_count = len(set(labels))
    if max_samples >= unique_label_count:
        _, sampled = train_test_split(
            examples,
            test_size=max_samples,
            random_state=seed,
            stratify=labels,
        )
        return list(sampled)

    randomizer = random.Random(seed)
    return randomizer.sample(examples, max_samples)


def predict_and_save(
    model: Any,
    sequence: Any,
    examples: Sequence[DatasetExample],
    output_path: Path,
) -> dict[str, float]:
    probabilities = model.predict(sequence, verbose=0)
    predicted_ids = probabilities.argmax(axis=1)
    true_ids = np.asarray([example.label_id for example in examples], dtype=np.int64)
    macro_f1 = float(f1_score(true_ids, predicted_ids, average="macro"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filepath",
        "split",
        "label_name",
        "label_id",
        "predicted_label_id",
        "predicted_label_name",
        "confidence",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for example, predicted_id, probability in zip(examples, predicted_ids, probabilities):
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

    return {"macro_f1": macro_f1}


def plot_training_history(history: dict[str, list[float]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history.get("loss", []), label="train")
    axes[0].plot(history.get("val_loss", []), label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.get("accuracy", []), label="train")
    axes[1].plot(history.get("val_accuracy", []), label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_model_summary(model: Any, output_path: Path) -> None:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


RESULT_FIELD_ORDER = [
    "model_id",
    "parameter_sharing",
    "parameter_count",
    "validation_macro_f1",
    "test_macro_f1",
    "train_samples",
    "validation_samples",
    "test_samples",
    "epochs",
    "image_height",
    "image_width",
    "batch_size",
]


def write_results_csv(results: Sequence[dict[str, object]], output_path: Path) -> None:
    if not results:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ordered_fieldnames(results)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalize_result_row(row, fieldnames) for row in results)


def ordered_fieldnames(results: Sequence[dict[str, object]]) -> list[str]:
    observed_fields: list[str] = []
    for row in results:
        for field in row:
            if field not in observed_fields:
                observed_fields.append(field)

    ordered = [field for field in RESULT_FIELD_ORDER if field in observed_fields]
    ordered.extend(field for field in observed_fields if field not in ordered)
    return ordered


def normalize_result_row(row: dict[str, object], fieldnames: Sequence[str]) -> dict[str, object]:
    return {field: row.get(field, "") for field in fieldnames}


def aggregate_shared_conv2d_results(artifacts_dir: Path) -> list[dict[str, object]]:
    valid_ids = {config.model_id for config in generate_required_cnn_configs()}
    required_metric_fields = {
        "parameter_count",
        "validation_macro_f1",
        "test_macro_f1",
        "train_samples",
        "validation_samples",
        "test_samples",
        "epochs",
        "image_height",
        "image_width",
        "batch_size",
    }
    results: list[dict[str, object]] = []
    for path in sorted((artifacts_dir / "metrics").glob("*.json")):
        payload = read_json(path)
        if (
            payload.get("model_id") in valid_ids
            and payload.get("parameter_sharing", "shared") == "shared"
            and required_metric_fields.issubset(payload)
        ):
            payload.setdefault("parameter_sharing", "shared")
            results.append(payload)
    return sorted(results, key=lambda row: str(row["model_id"]))


def write_ranked_results(results: Sequence[dict[str, object]], artifacts_dir: Path) -> None:
    ranked = sorted(results, key=lambda row: float(row.get("test_macro_f1", -1)), reverse=True)
    write_results_csv(ranked, artifacts_dir / "metrics" / "ranked_results.csv")
    if ranked:
        write_json(artifacts_dir / "metrics" / "best_shared_conv2d.json", ranked[0])


def ensure_cnn_artifact_dirs(artifacts_dir: Path) -> None:
    for dirname in ("configs", "models", "histories", "predictions", "metrics", "plots"):
        (artifacts_dir / dirname).mkdir(parents=True, exist_ok=True)


def select_model_configs(config_id: str | None, smoke: bool) -> list[CNNModelConfig]:
    configs = generate_required_cnn_configs()
    if config_id is not None:
        configs = [config for config in configs if config.model_id == config_id]
        if not configs:
            available = ", ".join(config.model_id for config in generate_required_cnn_configs())
            raise ValueError(f"Unknown config id: {config_id}. Available configs: {available}")
    if smoke:
        return configs[:1]
    return configs


def config_with_model_id(config: CNNModelConfig, model_id: str) -> CNNModelConfig:
    return replace(config, model_id=model_id)


def _is_train_split(split: str) -> bool:
    return split.lower() in {"seg_train", "train", "training"}


def _is_validation_split(split: str) -> bool:
    return split.lower() in {"seg_val", "seg_valid", "val", "valid", "validation"}


def _is_test_split(split: str) -> bool:
    return split.lower() in {"seg_test", "test", "testing"}


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
    parser = argparse.ArgumentParser(description="Train required Keras CNN architectures.")
    parser.add_argument(
        "--dataset-index",
        type=Path,
        default=root / "artifacts" / "cnn" / "dataset_index.csv",
        help="CSV dataset index generated by cnn.preprocessing.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=root / "artifacts" / "cnn",
        help="Root output directory for CNN training artifacts.",
    )
    parser.add_argument("--image-size", type=_parse_size, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config-id", help="Train only one generated architecture config.")
    parser.add_argument("--smoke", action="store_true", help="Run one tiny 1-epoch training job.")
    parser.add_argument("--force", action="store_true", help="Retrain configs even when artifacts exist.")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    args = parser.parse_args(argv)

    if args.smoke:
        args.epochs = 1
        args.max_train_samples = args.max_train_samples or 96
        args.max_validation_samples = args.max_validation_samples or 32
        args.max_test_samples = args.max_test_samples or 32

    run_config = TrainingRunConfig(
        dataset_index=args.dataset_index,
        artifacts_dir=args.artifacts_dir,
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
    model_configs = select_model_configs(args.config_id, args.smoke)
    results = train_required_cnn_configs(run_config, model_configs=model_configs)
    print(f"Completed {len(results)} CNN training run(s).")
    print(f"Results: {run_config.artifacts_dir / 'metrics' / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
