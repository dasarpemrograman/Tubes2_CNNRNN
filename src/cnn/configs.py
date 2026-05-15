from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


INTEL_CLASS_NAMES = ("buildings", "forest", "glacier", "mountain", "sea", "street")
DEFAULT_IMAGE_SIZE = (150, 150)
DEFAULT_BATCH_SIZE = 32


@dataclass(frozen=True)
class CNNPreprocessingConfig:
    dataset_root: Path
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
    batch_size: int = DEFAULT_BATCH_SIZE

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        if self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise ValueError("image_size must contain positive height and width.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")


PoolingType = Literal["max", "average"]
ClassifierPooling = Literal["flatten", "global_average", "global_max"]


@dataclass(frozen=True)
class CNNModelConfig:
    model_id: str
    conv_layers: int
    filters: tuple[int, ...]
    kernel_sizes: tuple[int, ...]
    pooling_type: PoolingType
    classifier_pooling: ClassifierPooling = "flatten"
    dense_units: int = 128
    dropout_rate: float = 0.3
    learning_rate: float = 0.001

    def __post_init__(self) -> None:
        if self.conv_layers <= 0:
            raise ValueError("conv_layers must be positive.")
        if len(self.filters) < self.conv_layers:
            raise ValueError("filters must provide at least conv_layers values.")
        if len(self.kernel_sizes) < self.conv_layers:
            raise ValueError("kernel_sizes must provide at least conv_layers values.")
        if any(value <= 0 for value in self.filters):
            raise ValueError("all filter counts must be positive.")
        if any(value <= 0 for value in self.kernel_sizes):
            raise ValueError("all kernel sizes must be positive.")
        if self.pooling_type not in {"max", "average"}:
            raise ValueError("pooling_type must be 'max' or 'average'.")
        if self.classifier_pooling not in {"flatten", "global_average", "global_max"}:
            raise ValueError(
                "classifier_pooling must be 'flatten', 'global_average', or 'global_max'."
            )
        if self.dense_units <= 0:
            raise ValueError("dense_units must be positive.")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout_rate must be in [0, 1).")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

    def as_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "conv_layers": self.conv_layers,
            "filters": list(self.filters[: self.conv_layers]),
            "kernel_sizes": list(self.kernel_sizes[: self.conv_layers]),
            "pooling_type": self.pooling_type,
            "classifier_pooling": self.classifier_pooling,
            "dense_units": self.dense_units,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
        }


def generate_required_cnn_configs() -> list[CNNModelConfig]:
    conv_layer_variants = (2, 3)
    filter_variants = ((32, 64, 128), (64, 128, 256))
    kernel_variants = ((3, 3, 3), (5, 3, 3))
    pooling_variants: tuple[PoolingType, ...] = ("max", "average")

    configs: list[CNNModelConfig] = []
    for conv_layers in conv_layer_variants:
        for filters in filter_variants:
            for kernel_sizes in kernel_variants:
                for pooling_type in pooling_variants:
                    active_filters = filters[:conv_layers]
                    active_kernels = kernel_sizes[:conv_layers]
                    model_id = (
                        f"cnn_c{conv_layers}"
                        f"_f{'-'.join(str(value) for value in active_filters)}"
                        f"_k{'-'.join(str(value) for value in active_kernels)}"
                        f"_{pooling_type}"
                    )
                    configs.append(
                        CNNModelConfig(
                            model_id=model_id,
                            conv_layers=conv_layers,
                            filters=filters,
                            kernel_sizes=kernel_sizes,
                            pooling_type=pooling_type,
                        )
                    )
    return configs
