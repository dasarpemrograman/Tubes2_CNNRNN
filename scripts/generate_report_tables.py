from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from common.paths import COMMON_ARTIFACTS_DIR, ensure_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert result CSV/JSON files into copyable tables.")
    parser.add_argument("inputs", nargs="*", type=Path, help="CSV or JSON result files.")
    parser.add_argument("--output", type=Path, default=COMMON_ARTIFACTS_DIR / "report_tables.md")
    args = parser.parse_args()

    ensure_dir(args.output.parent)
    sections: list[str] = []
    for input_path in args.inputs:
        if input_path.suffix.lower() == ".csv":
            frame = pd.read_csv(input_path)
        elif input_path.suffix.lower() == ".json":
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            frame = pd.DataFrame(payload if isinstance(payload, list) else [payload])
        else:
            raise ValueError(f"unsupported table input: {input_path}")

        sections.append(f"## {input_path.name}\n\n{frame.to_markdown(index=False)}")

    args.output.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
