#!/usr/bin/env python3
"""Train a YOLOv8 model for License Plate Number Recognition (LNPR).

Examples:
  python scripts/train.py --data data/lnpr_dataset/data.yaml --model yolov8n.pt --epochs 50
  python scripts/train.py --data data/lnpr_dataset/data.yaml --model yolov8s.pt --device cpu
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for LNPR")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Model name or path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="cuda device index, 'cpu', or None")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="lnpr")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cfg", default=None, help="Optional Ultralytics config yaml")
    return parser.parse_args()


def train_model(
    data: str,
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    project: str,
    name: str,
    resume: bool = False,
    device: str | None = None,
    cfg: str | None = None,
) -> None:
    from ultralytics import YOLO

    model = YOLO(model_name)
    train_kwargs = {
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": project,
        "name": name,
        "resume": resume,
    }

    if device is not None:
        train_kwargs["device"] = device

    if cfg:
        train_kwargs["cfg"] = cfg

    model.train(**train_kwargs)


def main() -> None:
    args = parse_args()
    train_model(
        data=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        resume=args.resume,
        device=args.device,
        cfg=args.cfg,
    )


if __name__ == "__main__":
    main()
