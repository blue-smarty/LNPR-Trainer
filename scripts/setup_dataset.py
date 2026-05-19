#!/usr/bin/env python3
"""Create a YOLOv8 dataset structure and data.yaml for LNPR training.

Usage:
  python scripts/setup_dataset.py --root data/lnpr_dataset --classes license_plate
"""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create YOLOv8 dataset structure for LNPR")
    parser.add_argument("--root", required=True, help="Root dataset directory")
    parser.add_argument(
        "--classes",
        default="license_plate",
        help="Comma-separated class names (default: license_plate)",
    )
    return parser.parse_args()


def setup_dataset(root_path: str, classes_csv: str) -> Path:
    root = Path(root_path).resolve()

    # Create standard YOLO directory structure
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    classes = [c.strip() for c in classes_csv.split(",") if c.strip()]
    if not classes:
        raise SystemExit("No classes provided. Use --classes license_plate")

    data = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(classes)},
    }

    data_path = root / "data.yaml"
    with data_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    return data_path


def main() -> None:
    args = parse_args()
    data_path = setup_dataset(root_path=args.root, classes_csv=args.classes)
    root = Path(args.root).resolve()

    print(f"Created dataset structure at: {root}")
    print(f"Wrote data.yaml: {data_path}")


if __name__ == "__main__":
    main()
