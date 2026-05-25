#!/usr/bin/env python3
"""Run end-to-end plate detection + CRNN text recognition on one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO


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
    parser = argparse.ArgumentParser(description="Detect and recognize plate text in one image")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--detector", required=True, help="Detector weights (.pt)")
    parser.add_argument("--recognizer", required=True, help="Recognizer checkpoint (.pt)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=10)
    parser.add_argument("--rec-img-height", type=int, default=32)
    parser.add_argument("--rec-img-width", type=int, default=160)
    parser.add_argument("--device", default="", help="cuda device index, 'cpu', or empty for auto")
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg.strip():
        return torch.device(device_arg.strip())
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_recognizer(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[CRNNRecognizer, str]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported recognizer checkpoint format: {checkpoint_path}")

    charset = str(ckpt.get("charset", ""))
    if not charset:
        raise RuntimeError("Recognizer checkpoint missing non-empty 'charset'.")

    state = ckpt.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("Recognizer checkpoint missing 'model_state_dict'.")

    rnn_hidden_size = int(ckpt.get("rnn_hidden_size", 256))
    model = CRNNRecognizer(num_classes=len(charset) + 1, rnn_hidden_size=rnn_hidden_size)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, charset


def preprocess_crop(crop_bgr: torch.Tensor | None, img_height: int, img_width: int) -> torch.Tensor:
    if crop_bgr is None:
        raise RuntimeError("Empty crop image")
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    return tensor


def decode_prediction(logits: torch.Tensor, charset: str) -> tuple[str, float]:
    blank_idx = len(charset)
    probs = logits.softmax(dim=2)[:, 0, :]
    argmax = probs.argmax(dim=1).tolist()

    chars: list[str] = []
    char_probs: list[float] = []
    prev = -1
    for t, idx in enumerate(argmax):
        if idx == blank_idx:
            prev = idx
            continue
        if idx == prev:
            continue
        chars.append(charset[idx])
        char_probs.append(float(probs[t, idx].item()))
        prev = idx

    conf = sum(char_probs) / len(char_probs) if char_probs else 0.0
    return "".join(chars), conf


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).resolve()
    detector_path = Path(args.detector).resolve()
    recognizer_path = Path(args.recognizer).resolve()

    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")
    if not detector_path.exists():
        raise SystemExit(f"Detector weights not found: {detector_path}")
    if not recognizer_path.exists():
        raise SystemExit(f"Recognizer checkpoint not found: {recognizer_path}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise SystemExit(f"Could not read image: {image_path}")

    device = choose_device(args.device)
    detector_device: str | int = "cpu"
    if device.type == "cuda":
        detector_device = 0 if device.index is None else device.index

    detector = YOLO(str(detector_path))
    rec_model, charset = load_recognizer(recognizer_path, device)

    preds = detector.predict(
        source=str(image_path),
        imgsz=int(args.imgsz),
        conf=float(args.conf),
        iou=float(args.iou),
        max_det=int(args.max_det),
        device=detector_device,
        verbose=False,
    )
    if not preds:
        payload = {"image": str(image_path), "num_detections": 0, "detections": []}
        text = json.dumps(payload, indent=2)
        print(text)
        if args.output_json.strip():
            output_json = Path(args.output_json).resolve()
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(text + "\n", encoding="utf-8")
        return

    result = preds[0]
    boxes = result.boxes

    detections: list[dict] = []
    crop_dir: Path | None = None
    if args.save_crops:
        crop_dir = image_path.parent / f"{image_path.stem}_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)

    if boxes is not None and boxes.xyxy is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [0.0] * len(xyxy)

        height, width = image_bgr.shape[:2]
        for i, (box, det_conf) in enumerate(zip(xyxy, confs, strict=False)):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image_bgr[y1:y2, x1:x2]
            crop_tensor = preprocess_crop(crop, int(args.rec_img_height), int(args.rec_img_width)).to(device)

            with torch.no_grad():
                logits = rec_model(crop_tensor)
            text, rec_conf = decode_prediction(logits, charset)

            crop_path = ""
            if crop_dir is not None:
                crop_file = crop_dir / f"crop_{i:03d}.png"
                cv2.imwrite(str(crop_file), crop)
                crop_path = str(crop_file)

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "detector_confidence": float(det_conf),
                    "text": text,
                    "recognizer_confidence": float(rec_conf),
                    "crop_path": crop_path,
                }
            )

    detections.sort(key=lambda d: d["bbox"][0])
    payload = {
        "image": str(image_path),
        "num_detections": len(detections),
        "detections": detections,
    }

    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    print(rendered)

    if args.output_json.strip():
        output_json = Path(args.output_json).resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
