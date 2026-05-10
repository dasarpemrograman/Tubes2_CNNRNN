from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.paths import ARTIFACTS_DIR, DOC_DIR, PROJECT_ROOT, REPORT_PDF

REQUIRED_ROOTS = [
    Path("src/common"),
    Path("src/cnn"),
    Path("src/captioning"),
    Path("src/scratch"),
    Path("src/experiments"),
    Path("src/notebooks"),
    Path("artifacts/cnn"),
    Path("artifacts/captioning"),
    Path("artifacts/common"),
    Path("artifacts/bonus"),
    Path("scripts"),
]

REQUIRED_PROJECT_FILES = [
    Path("README.md"),
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path(".python-version"),
    Path(".gitignore"),
]


def _missing(paths: list[Path]) -> list[Path]:
    return [path for path in paths if not (PROJECT_ROOT / path).exists()]


def audit_doc_policy() -> list[str]:
    issues: list[str] = []
    if not DOC_DIR.exists():
        issues.append("doc/ directory is missing")
        return issues

    entries = sorted(path.name for path in DOC_DIR.iterdir())
    if entries != ["report.pdf"]:
        issues.append(f"doc/ must contain exactly report.pdf; found {entries}")
    if not REPORT_PDF.exists():
        issues.append("doc/report.pdf is missing")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit repository and artifact structure.")
    parser.add_argument("--strict-artifacts", action="store_true", help="Require result files too.")
    args = parser.parse_args()

    issues = [f"missing required path: {path}" for path in _missing(REQUIRED_ROOTS)]
    issues.extend(f"missing required file: {path}" for path in _missing(REQUIRED_PROJECT_FILES))
    issues.extend(audit_doc_policy())

    if args.strict_artifacts:
        for path in [ARTIFACTS_DIR / "cnn", ARTIFACTS_DIR / "captioning"]:
            if not any(path.iterdir()):
                issues.append(f"{path.relative_to(PROJECT_ROOT)} has no result artifacts yet")

    if issues:
        print("Artifact audit failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("Artifact audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
