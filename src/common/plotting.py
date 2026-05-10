from __future__ import annotations

from pathlib import Path

from common.paths import ensure_dir

DEFAULT_FIGSIZE = (8, 5)
DEFAULT_DPI = 150


def save_figure(fig, path: Path) -> None:
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=DEFAULT_DPI, bbox_inches="tight")
