#!/usr/bin/env python3
"""Export a trained CRNN recognizer checkpoint to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


class CRNNRecognizer(nn.Module):
    def __init__(self, num_classes: int, rnn_hidden_size: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(512, 512, kernel_size=2),
            nn.ReLU(inplace=True),
        )
        self.sequence_pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=rnn_hidden_size,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
        )
        self.classifier = nn.Linear(rnn_hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.sequence_pool(feat).squeeze(2)
        seq = feat.permute(2, 0, 1).contiguous()
        seq, _ = self.rnn(seq)
        return self.classifier(seq)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CRNN recognizer checkpoint to ONNX")
    parser.add_argument("--weights", required=True, help="Path to recognizer checkpoint (.pt)")
    parser.add_argument("--img-height", type=int, default=32)
    parser.add_argument("--img-width", type=int, default=160)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="", help="Optional output .onnx path")
    parser.add_argument("--dynamic-width", action="store_true")
    return parser.parse_args()


def load_checkpoint(weights: Path, device: torch.device) -> dict:
    ckpt = torch.load(weights, map_location=device)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint format in {weights}")
    if "model_state_dict" not in ckpt:
        raise RuntimeError("Checkpoint missing 'model_state_dict'.")
    if "charset" not in ckpt:
        raise RuntimeError("Checkpoint missing 'charset'.")
    return ckpt


def main() -> None:
    args = parse_args()

    weights = Path(args.weights).resolve()
    if not weights.exists():
        raise SystemExit(f"Checkpoint not found: {weights}")

    output = Path(args.output).resolve() if args.output.strip() else weights.with_suffix(".onnx")
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    ckpt = load_checkpoint(weights, device)
    charset: str = str(ckpt["charset"])
    rnn_hidden_size = int(ckpt.get("rnn_hidden_size", 256))

    model = CRNNRecognizer(num_classes=len(charset) + 1, rnn_hidden_size=rnn_hidden_size)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    dummy = torch.randn(1, 1, int(args.img_height), int(args.img_width), device=device)

    dynamic_axes = None
    if args.dynamic_width:
        dynamic_axes = {
            "input": {3: "width"},
            "logits": {0: "time_steps"},
        }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output),
            export_params=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=int(args.opset),
            do_constant_folding=True,
        )

    print(f"Exported ONNX to: {output}")


if __name__ == "__main__":
    main()
