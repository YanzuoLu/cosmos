#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)
    for pattern in ("summary.json", "frames/frame_metrics.json"):
        for path in src.glob(f"*/*/{pattern}"):
            rel = path.relative_to(src)
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst / rel)
    for path in src.glob("*/*/frames/pairs/*.png"):
        rel = path.relative_to(src)
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst / rel)
    aggregate = src / "aggregate_summary.json"
    if aggregate.exists():
        shutil.copy2(aggregate, dst / "aggregate_summary.json")
    print(f"staged {dst}")


if __name__ == "__main__":
    main()
