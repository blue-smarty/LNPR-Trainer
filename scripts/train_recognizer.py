#!/usr/bin/env python3
"""Train a CRNN recognizer for license plate text.

Expected dataset layout:
  <root>/train/images/*
  <root>/val/images/*
  <root>/train/labels.txt (or compatible alternatives)
  <root>/val/labels.txt (or compatible alternatives)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class Sample:
    image_path: Path
    text: str


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
        feat = self.sequence_pool(feat).squeeze(2)  # (B, C, W)
        seq = feat.permute(2, 0, 1).contiguous()  # (T, B, C)
        seq, _ = self.rnn(seq)
        return self.classifier(seq)  # (T, B, C)


class OCRDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(
        self,
        samples: list[Sample],
        charset_to_index: dict[str, int],
        img_height: int,
        img_width: int,
    ) -> None:
        self.samples = samples
        self.charset_to_index = charset_to_index
        self.img_height = img_height
        self.img_width = img_width

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[idx]
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {sample.image_path}")

        resized = cv2.resize(
            image,
            (self.img_width, self.img_height),
            interpolation=cv2.INTER_LINEAR,
        )
        image_tensor = torch.from_numpy(resized).float().unsqueeze(0) / 255.0
        encoded = torch.tensor(
            [self.charset_to_index[ch] for ch in sample.text],
            dtype=torch.long,
        )
        return image_tensor, encoded, sample.text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CRNN plate-text recognizer")
    parser.add_argument("--data", required=True, help="Recognition dataset root directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--img-height", type=int, default=32)
    parser.add_argument("--img-width", type=int, default=160)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--charset",
        default="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        help="Allowed recognition characters",
    )
    parser.add_argument("--rnn-hidden-size", type=int, default=256)
    parser.add_argument("--project", default="runs/recognition")
    parser.add_argument("--name", default="crnn_lp")
    parser.add_argument(
        "--device",
        default="",
        help="device: '', 'cpu', 'cuda', 'cuda:N', or numeric CUDA index like '0'",
    )
    return parser.parse_args()


def dedupe_charset(charset: str) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for ch in charset:
        if ch in seen:
            continue
        seen.add(ch)
        out.append(ch)
    return "".join(out)


def parse_label_line(line: str) -> tuple[str, str] | None:
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None

    for sep in ("\t", ","):
        if sep in raw:
            left, right = raw.split(sep, 1)
            left = left.strip()
            right = right.strip()
            if left and right:
                return left, right

    parts = raw.split(maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()

    return None


def candidate_label_files(root: Path, split: str) -> list[Path]:
    return [
        root / split / "labels.txt",
        root / f"{split}.txt",
        root / f"labels_{split}.txt",
        root / "labels.txt",
    ]


def resolve_image_path(image_ref: str, labels_file: Path, split_images_dir: Path) -> Path:
    candidate = Path(image_ref)
    if candidate.is_absolute():
        return candidate

    from_labels = (labels_file.parent / candidate).resolve()
    if from_labels.exists():
        return from_labels

    from_split = (split_images_dir / candidate.name).resolve()
    if from_split.exists():
        return from_split

    return from_labels


def load_split_samples(root: Path, split: str, charset: set[str]) -> list[Sample]:
    split_images_dir = root / split / "images"
    labels_file = next((p for p in candidate_label_files(root, split) if p.exists()), None)
    if labels_file is None:
        raise FileNotFoundError(
            f"No labels file found for split '{split}'. Tried: "
            + ", ".join(str(p) for p in candidate_label_files(root, split))
        )

    if not split_images_dir.exists():
        raise FileNotFoundError(f"Missing images directory for split '{split}': {split_images_dir}")

    samples: list[Sample] = []
    with labels_file.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_label_line(line)
            if parsed is None:
                continue

            image_ref, text = parsed
            text = text.strip()
            if not text or any(ch not in charset for ch in text):
                continue

            image_path = resolve_image_path(image_ref, labels_file, split_images_dir)
            if not image_path.exists():
                continue

            samples.append(Sample(image_path=image_path, text=text))

    if not samples:
        raise RuntimeError(
            f"No valid samples loaded for split '{split}' from labels file: {labels_file}"
        )

    return samples


def collate_fn(
    batch: Iterable[tuple[torch.Tensor, torch.Tensor, str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    images: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    target_lengths: list[int] = []
    texts: list[str] = []

    for image, encoded, text in batch:
        images.append(image)
        targets.append(encoded)
        target_lengths.append(int(encoded.numel()))
        texts.append(text)

    images_tensor = torch.stack(images, dim=0)
    targets_tensor = torch.cat(targets, dim=0) if targets else torch.tensor([], dtype=torch.long)
    target_lengths_tensor = torch.tensor(target_lengths, dtype=torch.long)
    return images_tensor, targets_tensor, target_lengths_tensor, texts


def decode_greedy(logits: torch.Tensor, charset: str, blank_idx: int) -> list[str]:
    indices = logits.argmax(dim=2).transpose(0, 1).cpu().tolist()  # (B, T)
    decoded: list[str] = []
    for seq in indices:
        out: list[str] = []
        prev = -1
        for idx in seq:
            if idx == blank_idx:
                prev = idx
                continue
            if idx == prev:
                continue
            out.append(charset[idx])
            prev = idx
        decoded.append("".join(out))
    return decoded


def choose_device(device_arg: str) -> torch.device:
    raw = device_arg.strip()
    if raw:
        candidate = f"cuda:{raw}" if raw.isdigit() else raw
        try:
            return torch.device(candidate)
        except (TypeError, RuntimeError, ValueError) as exc:
            raise SystemExit(f"Invalid --device value '{device_arg}': {exc}") from exc
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_recognizer(args: argparse.Namespace) -> None:
    charset = dedupe_charset(args.charset)
    if not charset:
        raise SystemExit("--charset must include at least one character")

    root = Path(args.data).resolve()
    charset_to_index = {ch: i for i, ch in enumerate(charset)}
    blank_idx = len(charset)

    train_samples = load_split_samples(root, "train", set(charset))
    val_samples = load_split_samples(root, "val", set(charset))

    train_ds = OCRDataset(train_samples, charset_to_index, args.img_height, args.img_width)
    val_ds = OCRDataset(val_samples, charset_to_index, args.img_height, args.img_width)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    device = choose_device(args.device)
    model = CRNNRecognizer(num_classes=len(charset) + 1, rnn_hidden_size=args.rnn_hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CTCLoss(blank=blank_idx, reduction="mean", zero_infinity=True)

    run_dir = Path(args.project).resolve() / args.name
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last.pt"

    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for images, targets, target_lengths, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            logits = model(images)
            log_probs = logits.log_softmax(dim=2)
            input_lengths = torch.full(
                (images.size(0),),
                fill_value=logits.size(0),
                dtype=torch.long,
                device=device,
            )

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.detach().cpu())
            train_batches += 1

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets, target_lengths, texts in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)

                logits = model(images)
                log_probs = logits.log_softmax(dim=2)
                input_lengths = torch.full(
                    (images.size(0),),
                    fill_value=logits.size(0),
                    dtype=torch.long,
                    device=device,
                )
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

                predictions = decode_greedy(logits, charset, blank_idx)
                for pred, truth in zip(predictions, texts):
                    correct += int(pred == truth)
                    total += 1

                val_loss_sum += float(loss.detach().cpu())
                val_batches += 1

        train_loss = train_loss_sum / max(1, train_batches)
        val_loss = val_loss_sum / max(1, val_batches)
        val_acc = correct / max(1, total)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "charset": charset,
            "img_height": int(args.img_height),
            "img_width": int(args.img_width),
            "rnn_hidden_size": int(args.rnn_hidden_size),
            "val_acc": float(val_acc),
        }
        torch.save(checkpoint, last_path)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, best_path)

        print(
            f"epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    print(f"Saved best checkpoint: {best_path}")
    print(f"Saved last checkpoint: {last_path}")


def main() -> None:
    args = parse_args()
    train_recognizer(args)


if __name__ == "__main__":
    main()
