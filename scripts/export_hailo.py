#!/usr/bin/env python3
"""Export a trained LNPR YOLOv8 model to ONNX for Hailo compilation.

Example:
  python scripts/export_hailo.py --weights runs/detect/lnpr/weights/best.pt --imgsz 640
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLOv8 LNPR model to ONNX")
    parser.add_argument("--weights", required=True, help="Path to trained weights")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic shapes")
    return parser.parse_args()


def export_onnx(
    weights: str,
    imgsz: int = 640,
    batch: int = 1,
    opset: int = 12,
    dynamic: bool = False,
) -> None:
    from ultralytics import YOLO

    model = YOLO(weights)
    model.export(
        format="onnx",
        imgsz=imgsz,
        batch=batch,
        opset=opset,
        dynamic=dynamic,
        simplify=True,
    )


def main() -> None:
    args = parse_args()
    export_onnx(
        weights=args.weights,
        imgsz=args.imgsz,
        batch=args.batch,
        opset=args.opset,
        dynamic=args.dynamic,
    )


if __name__ == "__main__":
    main()
