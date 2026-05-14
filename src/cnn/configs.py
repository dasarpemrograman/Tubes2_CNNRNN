from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
