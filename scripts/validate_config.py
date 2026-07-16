"""Lightweight YAML config validator for LNPR-Trainer.

Usage:
    python scripts/validate_config.py <config_path>
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REQUIRED_TOP_LEVEL_KEYS = {"seed", "output_dir", "device", "data", "training"}
REQUIRED_DATA_KEYS = {"train_path", "val_path", "text_column", "label_column"}
REQUIRED_TRAINING_KEYS = {
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "early_stopping_patience",
    "checkpoint_dir",
}


def validate_config(config_path: str | Path) -> None:
    """Validate the YAML config at *config_path*.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
        ValueError: If required keys are missing from the config.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        config: dict = yaml.safe_load(fh) or {}

    missing_top = REQUIRED_TOP_LEVEL_KEYS - config.keys()
    if missing_top:
        raise ValueError(f"Missing required top-level keys: {sorted(missing_top)}")

    data_section = config.get("data", {}) or {}
    missing_data = REQUIRED_DATA_KEYS - data_section.keys()
    if missing_data:
        raise ValueError(f"Missing required 'data' keys: {sorted(missing_data)}")

    training_section = config.get("training", {}) or {}
    missing_training = REQUIRED_TRAINING_KEYS - training_section.keys()
    if missing_training:
        raise ValueError(f"Missing required 'training' keys: {sorted(missing_training)}")

    print(f"Config OK: {path}")


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: python {Path(__file__).name} <config_path>", file=sys.stderr)
        sys.exit(1)

    try:
        validate_config(sys.argv[1])
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
