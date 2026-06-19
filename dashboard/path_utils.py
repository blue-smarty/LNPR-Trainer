"""Dataset path resolution utilities for LNPR Trainer.

This module is intentionally free of Streamlit imports so its functions can
be exercised by unit tests without needing a Streamlit installation.
"""

from __future__ import annotations

from pathlib import Path
import urllib.request
import zipfile

# Repository root: two directories above this file (dashboard/path_utils.py).
REPO_ROOT: Path = Path(__file__).resolve().parents[1]


def resolve_dataset_path_or_url(path_or_url: str, tmp_dir: Path) -> Path:
    """Resolve a local path or http(s) URL to a ``data.yaml`` path.

    Supported inputs:

    * http(s) URL to a ``.zip`` archive — downloaded and extracted.
    * http(s) URL directly to a YAML file — downloaded as-is.
    * Local ``.zip`` file — extracted; ``data.yaml`` located inside.
    * Local directory — ``data.yaml`` is expected inside the directory.
    * Local ``data.yaml`` file — returned as-is.

    Relative local paths are always resolved against :data:`REPO_ROOT` (the
    repository root), **not** against the process working directory.  This
    prevents duplicated path segments such as
    ``…/dashboard/dashboard/data/lnpr_dataset/data.yaml`` that would arise
    when Streamlit is launched from within the ``dashboard/`` sub-directory.

    Raises ``FileNotFoundError`` when the path or extracted content contains no
    ``data.yaml``.  Raises ``ValueError`` for unrecognised input.
    """
    p = path_or_url.strip()
    is_url = p.startswith("http://") or p.startswith("https://")

    if is_url:
        url_lower = p.lower().split("?")[0]  # strip query params for suffix check
        if url_lower.endswith(".zip"):
            dl_path = tmp_dir / "dataset_url.zip"
            urllib.request.urlretrieve(p, dl_path)  # noqa: S310
            extract_dir = tmp_dir / "dataset_url"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(dl_path, "r") as zf:
                extract_root = extract_dir.resolve()
                for member in zf.infolist():
                    member_path = (extract_dir / member.filename).resolve()
                    if not str(member_path).startswith(str(extract_root)):
                        raise ValueError(
                            f"Unsafe path in archive member: {member.filename}"
                        )
                zf.extractall(extract_dir)
            yaml_files = sorted(extract_dir.rglob("data.yaml"))
            if not yaml_files:
                raise FileNotFoundError(
                    f"No data.yaml found in the downloaded archive: {p}"
                )
            return yaml_files[0]
        else:
            # Treat as a direct link to a YAML/data.yaml file.
            dl_path = tmp_dir / "data.yaml"
            urllib.request.urlretrieve(p, dl_path)  # noqa: S310
            return dl_path

    # Local path — resolve relative paths against REPO_ROOT so that the result
    # is always absolute and independent of the process working directory.
    local = Path(p)
    if not local.is_absolute():
        local = (REPO_ROOT / p).resolve()

    if not local.exists():
        raise FileNotFoundError(f"Dataset path not found: {local}")

    if local.is_file():
        if local.suffix.lower() == ".zip":
            extract_dir = tmp_dir / "dataset_local"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(local, "r") as zf:
                extract_root = extract_dir.resolve()
                for member in zf.infolist():
                    member_path = (extract_dir / member.filename).resolve()
                    if not str(member_path).startswith(str(extract_root)):
                        raise ValueError(
                            f"Unsafe path in archive member: {member.filename}"
                        )
                zf.extractall(extract_dir)
            yaml_files = sorted(extract_dir.rglob("data.yaml"))
            if not yaml_files:
                raise FileNotFoundError(
                    f"No data.yaml found in the archive: {local}"
                )
            return yaml_files[0]
        # Assume it is a data.yaml (or similar YAML config) file.
        return local

    if local.is_dir():
        yaml_path = local / "data.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"No data.yaml found in directory: {local}"
            )
        return yaml_path

    raise ValueError(f"Path is neither a file nor a directory: {local}")


def resolve_local_path(path_str: str, base: Path = REPO_ROOT) -> Path:
    """Return *path_str* as an absolute path resolved against *base*.

    If *path_str* is already absolute it is returned as-is (after
    ``Path.resolve()`` to normalise any ``..`` components).  Relative paths
    are joined onto *base* before resolving.

    This helper is the canonical way to normalise any user-supplied local path
    (e.g. a selectbox value or a text-input value) before use, ensuring the
    result is always independent of the process working directory.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (base / path_str).resolve()
