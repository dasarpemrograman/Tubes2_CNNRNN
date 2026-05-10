from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DOC_DIR = PROJECT_ROOT / "doc"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"

CNN_ARTIFACTS_DIR = ARTIFACTS_DIR / "cnn"
CAPTIONING_ARTIFACTS_DIR = ARTIFACTS_DIR / "captioning"
COMMON_ARTIFACTS_DIR = ARTIFACTS_DIR / "common"
BONUS_ARTIFACTS_DIR = ARTIFACTS_DIR / "bonus"

REPORT_PDF = DOC_DIR / "report.pdf"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
