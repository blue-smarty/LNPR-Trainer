"""Artifact discovery helpers for the LNPR Trainer dashboard."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunInfo:
    """Information about a single training run directory."""

    path: Path
    name: str
    mtime: float
    weights: list[Path] = field(default_factory=list)
    onnx_files: list[Path] = field(default_factory=list)
    hef_files: list[Path] = field(default_factory=list)

    @property
    def best_pt(self) -> Path | None:
        for p in self.weights:
            if p.name == "best.pt":
                return p
        return None

    @property
    def last_pt(self) -> Path | None:
        for p in self.weights:
            if p.name == "last.pt":
                return p
        return None


def find_recent_runs(runs_root: Path, max_runs: int = 10) -> list[RunInfo]:
    """Return the most-recent training run directories under *runs_root*.

    Scans ``runs_root`` (typically ``runs/detect``) for sub-directories that
    contain a ``weights/`` folder, then returns them sorted newest-first.
    """
    runs_root = Path(runs_root)
    if not runs_root.exists():
        return []

    infos: list[RunInfo] = []
    for candidate in runs_root.iterdir():
        if not candidate.is_dir():
            continue
        weights_dir = candidate / "weights"
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            continue

        weights: list[Path] = []
        if weights_dir.exists():
            weights = sorted(
                [p for p in weights_dir.iterdir() if p.suffix == ".pt"],
                key=lambda p: p.name,
            )

        onnx_files: list[Path] = sorted(
            candidate.rglob("*.onnx"), key=lambda p: p.stat().st_mtime
        )
        hef_files: list[Path] = sorted(
            candidate.rglob("*.hef"), key=lambda p: p.stat().st_mtime
        )

        infos.append(
            RunInfo(
                path=candidate,
                name=candidate.name,
                mtime=mtime,
                weights=weights,
                onnx_files=onnx_files,
                hef_files=hef_files,
            )
        )

    infos.sort(key=lambda r: r.mtime, reverse=True)
    return infos[:max_runs]


def find_all_onnx(repo_root: Path) -> list[Path]:
    """Return all ``.onnx`` files found anywhere under *repo_root*, newest first."""
    repo_root = Path(repo_root)
    onnx_files = list(repo_root.rglob("*.onnx"))
    onnx_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return onnx_files


def find_all_hef(repo_root: Path) -> list[Path]:
    """Return all ``.hef`` files found anywhere under *repo_root*, newest first."""
    repo_root = Path(repo_root)
    hef_files = list(repo_root.rglob("*.hef"))
    hef_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hef_files


def format_size(path: Path) -> str:
    """Return a human-readable file size string for *path*."""
    try:
        size = path.stat().st_size
    except OSError:
        return "unknown size"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size = size / 1024
    return f"{size:.1f} TB"


def format_mtime(path: Path) -> str:
    """Return a human-readable last-modified string for *path*."""
    import datetime

    try:
        ts = path.stat().st_mtime
    except OSError:
        return "unknown time"
    dt = datetime.datetime.fromtimestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M")


def infer_onnx_path(weights_path: Path) -> Path | None:
    """Guess the ONNX output path from a ``.pt`` weights path.

    Ultralytics saves the exported ONNX next to the weights file with the
    same stem and a ``.onnx`` suffix.
    """
    candidate = weights_path.with_suffix(".onnx")
    if candidate.exists():
        return candidate
    return None
