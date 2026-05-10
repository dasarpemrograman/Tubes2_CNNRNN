from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def validate_image(path: Path, min_width: int, min_height: int) -> str | None:
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    except Exception as exc:
        return f"{path}: unreadable image ({exc})"

    if width < min_width or height < min_height:
        return f"{path}: too small ({width}x{height}, expected >= {min_width}x{min_height})"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate report figures before Google Docs insertion.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--min-width", type=int, default=640)
    parser.add_argument("--min-height", type=int, default=360)
    args = parser.parse_args()

    files: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            files.extend(path.rglob("*.png"))
            files.extend(path.rglob("*.jpg"))
            files.extend(path.rglob("*.jpeg"))
        else:
            files.append(path)

    issues = [issue for file in files if (issue := validate_image(file, args.min_width, args.min_height))]
    if issues:
        print("Figure validation failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print(f"Validated {len(files)} figure(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
