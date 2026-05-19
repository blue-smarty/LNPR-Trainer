#!/usr/bin/env python3
"""Convert an ONNX model to a Hailo HEF file using the Hailo Dataflow Compiler.

Requires the Hailo SDK (``hailo_sdk_client``) to be installed.
Download it from the Hailo Developer Zone: https://developer.hailo.ai

Example:
  python scripts/onnx_to_hef.py --onnx runs/detect/lnpr/weights/best.onnx --hw-arch hailo8l
  python scripts/onnx_to_hef.py --onnx best.onnx --hw-arch hailo8 --calib-path data/calib_images
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Supported Hailo hardware architectures.  Kept here so that the dashboard
# validator can import the same tuple and avoid drift.
VALID_HW_ARCHS: tuple[str, ...] = ("hailo8", "hailo8l", "hailo8r")

# Number of randomly generated images used when no calibration directory is
# provided.  16 frames gives the compiler enough statistical diversity for a
# basic quantization pass while remaining fast.
_DEFAULT_RANDOM_CALIB_IMAGES = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_node_names(raw_nodes: list[str] | None) -> list[str]:
    """Normalize node names by trimming whitespace and dropping empties."""
    return [n.strip() for n in (raw_nodes or []) if n and n.strip()]


def _extract_suggested_end_nodes(error_text: str) -> list[str]:
    """Extract parser-suggested end nodes from Hailo error text."""
    match = re.search(
        r"using these end node names:\s*(.+?)(?:\n|$)",
        error_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    return _parse_node_names(match.group(1).split(","))


def _load_calibration_images(
    calib_path: str,
    height: int,
    width: int,
) -> "np.ndarray":
    """Load images from *calib_path*, resize to (*height*, *width*), and return
    a float32 NumPy array of shape ``[N, H, W, 3]`` normalised to ``[0, 1]``.
    """
    import numpy as np

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "opencv-python is required to load calibration images. "
            "Install it with: pip install opencv-python"
        ) from exc

    calib_dir = Path(calib_path)
    if not calib_dir.is_dir():
        raise NotADirectoryError(f"Calibration path is not a directory: {calib_dir}")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_paths = sorted(p for p in calib_dir.rglob("*") if p.suffix.lower() in image_exts)
    if not image_paths:
        raise FileNotFoundError(f"No images found in calibration directory: {calib_dir}")

    images: list[np.ndarray] = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, (width, height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        images.append(img)  # HWC

    if not images:
        raise ValueError(f"Could not decode any images from: {calib_dir}")

    return np.stack(images, axis=0)  # shape: [N, H, W, 3]


def _extract_expected_input_shape(error_text: str) -> tuple[int, ...] | None:
    """Extract expected network input shape from a Hailo mismatch error."""
    match = re.search(r"network's input shape\s*\(([^)]+)\)", error_text)
    if not match:
        return None
    try:
        return tuple(int(part.strip()) for part in match.group(1).split(","))
    except ValueError:
        return None


def _permute_calib_to_expected_shape(
    calib_data: "np.ndarray", expected_shape: tuple[int, ...]
) -> "np.ndarray | None":
    """Return calibration data permuted to expected sample shape when possible."""
    sample_shape = tuple(calib_data.shape[1:])
    if len(sample_shape) != len(expected_shape):
        return None
    if len(sample_shape) > 4:
        return None

    used = [False] * len(sample_shape)
    perm: list[int] = []
    for dim in expected_shape:
        match_idx = None
        for idx, sample_dim in enumerate(sample_shape):
            if not used[idx] and sample_dim == dim:
                match_idx = idx
                break
        if match_idx is None:
            return None
        used[match_idx] = True
        perm.append(match_idx)

    if tuple(sample_shape[i] for i in perm) != expected_shape:
        return None
    if perm == list(range(len(sample_shape))):
        return calib_data
    batch_first_axes = tuple([0, *[i + 1 for i in perm]])
    return calib_data.transpose(batch_first_axes)


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def convert_onnx_to_hef(
    onnx_path: str,
    output_dir: str | None = None,
    hw_arch: str = "hailo8l",
    calib_path: str | None = None,
    input_shape: tuple[int, int] = (640, 640),
    end_nodes: list[str] | None = None,
) -> Path:
    """Convert *onnx_path* to a Hailo HEF file.

    Parameters
    ----------
    onnx_path:
        Path to the input ``.onnx`` model.
    output_dir:
        Directory where the ``.hef`` file is written.  Defaults to the same
        directory as the ONNX file.
    hw_arch:
        Target Hailo hardware architecture, e.g. ``"hailo8l"`` or ``"hailo8"``.
    calib_path:
        Optional path to a directory of calibration images.  Providing real
        images produces more accurate INT8 quantization.  When omitted, random
        data is used as a fallback.
    input_shape:
        ``(height, width)`` of the model input (default ``(640, 640)``).
    end_nodes:
        Optional ONNX graph end node names to pass to the Hailo parser.

    Returns
    -------
    Path
        Path to the written ``.hef`` file.

    Raises
    ------
    ImportError
        When ``hailo_sdk_client`` is not installed.
    FileNotFoundError
        When *onnx_path* does not exist.
    """
    try:
        from hailo_sdk_client import ClientRunner
    except ImportError as exc:
        raise ImportError(
            "The Hailo SDK (hailo_sdk_client) is required for ONNX → HEF conversion.\n"
            "Download and install it from the Hailo Developer Zone:\n"
            "  https://developer.hailo.ai"
        ) from exc

    import numpy as np

    onnx_file = Path(onnx_path)
    if not onnx_file.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_file}")

    out_dir = Path(output_dir) if output_dir else onnx_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    hef_path = out_dir / onnx_file.with_suffix(".hef").name
    model_name = onnx_file.stem

    runner = ClientRunner(hw_arch=hw_arch)
    explicit_end_nodes = _parse_node_names(end_nodes)
    try:
        if explicit_end_nodes:
            runner.translate_onnx_model(
                str(onnx_file),
                model_name,
                end_node_names=explicit_end_nodes,
            )
        else:
            runner.translate_onnx_model(str(onnx_file), model_name)
    except Exception as exc:
        # Common Hailo parser failure mode: a parse error that includes suggested
        # end node names (e.g. "... using these end node names: /a, /b").
        if explicit_end_nodes:
            raise
        suggested_end_nodes = _extract_suggested_end_nodes(str(exc))
        if not suggested_end_nodes:
            raise
        print(
            "ONNX parser suggested end nodes; retrying with: "
            f"{', '.join(suggested_end_nodes)}"
        )
        runner.translate_onnx_model(
            str(onnx_file),
            model_name,
            end_node_names=suggested_end_nodes,
        )

    h, w = input_shape
    if calib_path:
        calib_data = _load_calibration_images(calib_path, h, w)
    else:
        # Use random calibration data as fallback — less accurate quantization.
        calib_data = np.random.rand(_DEFAULT_RANDOM_CALIB_IMAGES, h, w, 3).astype(np.float32)

    try:
        runner.optimize(calib_data)
    except Exception as exc:
        error_text = str(exc)
        if "doesn't match network's input shape" not in error_text:
            raise
        expected_shape = _extract_expected_input_shape(error_text)
        if not expected_shape:
            raise
        permuted = _permute_calib_to_expected_shape(calib_data, expected_shape)
        if permuted is None:
            raise
        print(
            "Calibration data shape mismatch detected; retrying optimization with "
            f"sample shape {tuple(permuted.shape[1:])}."
        )
        runner.optimize(permuted)

    hef_bytes = runner.compile()
    hef_path.write_bytes(hef_bytes)
    return hef_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an ONNX LNPR model to a Hailo HEF file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Requires the Hailo SDK (hailo_sdk_client).\n"
            "Download from https://developer.hailo.ai\n\n"
            "Examples:\n"
            "  python scripts/onnx_to_hef.py --onnx best.onnx --hw-arch hailo8l\n"
            "  python scripts/onnx_to_hef.py --onnx best.onnx --calib-path data/calib"
        ),
    )
    parser.add_argument(
        "--onnx",
        required=True,
        help="Path to the input .onnx file",
    )
    parser.add_argument(
        "--hw-arch",
        default="hailo8l",
        choices=list(VALID_HW_ARCHS),
        help="Target Hailo hardware architecture (default: hailo8l)",
    )
    parser.add_argument(
        "--calib-path",
        default=None,
        metavar="DIR",
        help="Directory of calibration images for INT8 quantization (recommended)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Output directory for the .hef file (default: same directory as .onnx)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Model input image size in pixels (default: 640)",
    )
    parser.add_argument(
        "--end-node",
        action="append",
        default=None,
        metavar="NODE",
        help=(
            "Optional ONNX end node name for Hailo parsing. "
            "Pass multiple times for multiple nodes."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hef_path = convert_onnx_to_hef(
        onnx_path=args.onnx,
        output_dir=args.output_dir,
        hw_arch=args.hw_arch,
        calib_path=args.calib_path,
        input_shape=(args.imgsz, args.imgsz),
        end_nodes=args.end_node,
    )
    print(f"HEF file written to: {hef_path}")


if __name__ == "__main__":
    main()
