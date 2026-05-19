"""Preflight validation helpers for the LNPR Trainer dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import yaml

from scripts.onnx_to_hef import VALID_HW_ARCHS


class ValidationResult(NamedTuple):
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def validate_setup_params(root_path: str, classes_csv: str) -> ValidationResult:
    """Validate dataset-setup parameters before running setup_dataset."""
    errors: list[str] = []
    warnings: list[str] = []

    if not root_path or not root_path.strip():
        errors.append("Dataset root path must not be empty.")

    raw_classes = [c.strip() for c in classes_csv.split(",")]
    non_empty = [c for c in raw_classes if c]
    if not non_empty:
        errors.append("At least one class name is required (comma-separated list).")
    else:
        seen: set[str] = set()
        duplicates: list[str] = []
        for cls in non_empty:
            if cls in seen:
                duplicates.append(cls)
            seen.add(cls)
        if duplicates:
            warnings.append(
                f"Duplicate class names detected and will be deduplicated: {', '.join(duplicates)}"
            )
        invalid = [c for c in non_empty if not c.replace("_", "").replace("-", "").isalnum()]
        if invalid:
            warnings.append(
                f"Class names with special characters may cause issues: {', '.join(invalid)}"
            )

    return ValidationResult(errors=errors, warnings=warnings)


def validate_train_params(
    data_yaml: str,
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    project: str,
    repo_root: Path | None = None,
) -> ValidationResult:
    """Validate training parameters before running train_model."""
    errors: list[str] = []
    warnings: list[str] = []

    if not data_yaml or not data_yaml.strip():
        errors.append("Path to data.yaml must not be empty.")
    else:
        data_path = Path(data_yaml)
        if repo_root and not data_path.is_absolute():
            data_path = (repo_root / data_yaml).resolve()
        if not data_path.exists():
            errors.append(f"data.yaml not found: {data_path}")
        else:
            _check_data_yaml(data_path, errors, warnings)

    if not model_name or not model_name.strip():
        errors.append("Model name must not be empty.")

    if epochs < 1:
        errors.append("Epochs must be at least 1.")
    elif epochs > 10000:
        warnings.append("Epoch count is unusually high (> 10,000).")

    if imgsz < 32:
        errors.append("Image size must be at least 32.")
    elif imgsz % 32 != 0:
        warnings.append(
            f"Image size {imgsz} is not a multiple of 32; YOLO typically requires multiples of 32."
        )

    if batch < 1:
        errors.append("Batch size must be at least 1.")

    if not project or not project.strip():
        errors.append("Project directory must not be empty.")

    return ValidationResult(errors=errors, warnings=warnings)


def _check_data_yaml(data_path: Path, errors: list[str], warnings: list[str]) -> None:
    """Inspect a data.yaml file for required keys and valid dataset paths."""
    try:
        with data_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        errors.append(f"Could not parse data.yaml: {exc}")
        return

    if not isinstance(data, dict):
        errors.append("data.yaml does not contain a valid YAML mapping.")
        return

    required_keys = ["path", "train", "val", "names"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        errors.append(f"data.yaml is missing required keys: {', '.join(missing)}")

    if "names" in data:
        names = data["names"]
        if not names:
            errors.append("data.yaml 'names' field is empty; at least one class is required.")
        elif not isinstance(names, dict):
            errors.append("data.yaml 'names' field should be a dict mapping index to name.")

    dataset_root = data.get("path")
    if dataset_root:
        root = Path(dataset_root)
        for split_key in ("train", "val"):
            rel = data.get(split_key)
            if rel:
                split_path = root / rel
                if not split_path.exists():
                    warnings.append(
                        f"Dataset split '{split_key}' path does not exist: {split_path}"
                    )


def validate_hef_params(
    onnx_path: str,
    hw_arch: str,
    calib_path: str,
    repo_root: Path | None = None,
) -> ValidationResult:
    """Validate parameters before running convert_onnx_to_hef."""
    errors: list[str] = []
    warnings: list[str] = []

    if not onnx_path or not onnx_path.strip():
        errors.append("ONNX file path must not be empty.")
    else:
        p = Path(onnx_path)
        if repo_root and not p.is_absolute():
            p = (repo_root / onnx_path).resolve()
        if not p.exists():
            errors.append(f"ONNX file not found: {p}")
        elif p.suffix.lower() != ".onnx":
            warnings.append(
                f"Selected file does not have a .onnx extension: '{p.name}'"
            )

    valid_archs = set(VALID_HW_ARCHS)
    if hw_arch not in valid_archs:
        errors.append(
            f"Unknown hardware architecture '{hw_arch}'. "
            f"Choose one of: {', '.join(sorted(valid_archs))}"
        )

    if calib_path and calib_path.strip():
        calib_dir = Path(calib_path)
        if repo_root and not calib_dir.is_absolute():
            calib_dir = (repo_root / calib_path).resolve()
        if not calib_dir.exists():
            errors.append(f"Calibration directory not found: {calib_dir}")
        elif not calib_dir.is_dir():
            errors.append(f"Calibration path is not a directory: {calib_dir}")

    return ValidationResult(errors=errors, warnings=warnings)


def validate_export_params(
    weights: str,
    imgsz: int,
    batch: int,
    opset: int,
    repo_root: Path | None = None,
) -> ValidationResult:
    """Validate export parameters before running export_onnx."""
    errors: list[str] = []
    warnings: list[str] = []

    if not weights or not weights.strip():
        errors.append("Weights path must not be empty.")
    else:
        weights_path = Path(weights)
        if repo_root and not weights_path.is_absolute():
            weights_path = (repo_root / weights).resolve()
        if not weights_path.exists():
            errors.append(f"Weights file not found: {weights_path}")
        elif weights_path.suffix.lower() != ".pt":
            warnings.append(
                f"Weights file does not have a .pt extension: '{weights_path.name}'"
            )

    if imgsz < 32:
        errors.append("Image size must be at least 32.")
    elif imgsz % 32 != 0:
        warnings.append(
            f"Image size {imgsz} is not a multiple of 32; YOLO typically requires multiples of 32."
        )

    if batch < 1:
        errors.append("Batch size must be at least 1.")

    if opset < 9:
        errors.append("ONNX opset version must be at least 9.")
    elif opset > 18:
        warnings.append(f"ONNX opset {opset} is newer than commonly supported versions (≤ 18).")

    return ValidationResult(errors=errors, warnings=warnings)
