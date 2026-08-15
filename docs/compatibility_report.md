# LNPR-Trainer Dependency Compatibility Report

Generated: 2026-08-15

## Summary

All required packages are importable and functional in the tested environment.
The Hailo SDK is an optional, proprietary dependency that requires separate
installation from the [Hailo Developer Zone](https://developer.hailo.ai).

---

## Python Version Requirements

| Requirement | Minimum | Tested |
|---|---|---|
| pyproject.toml `requires-python` | `>=3.10` | 3.12.3 ✅ |
| Raspberry Pi 5 baseline | `>=3.9` | 3.12.3 ✅ |

> **Note:** The code uses `str | None` union syntax (PEP 604) which requires
> Python 3.10+.  All scripts have `from __future__ import annotations` which
> makes this forward-compatible down to Python 3.7, but the union syntax in
> function signatures still requires 3.10+ at runtime without that import.
> Ensure `from __future__ import annotations` is present in every file if
> Raspberry Pi 5 ships with Python 3.9.

---

## Required Dependencies

| Package | `requirements.txt` | Tested Version | Status |
|---|---|---|---|
| ultralytics | `>=8.0.0` | 8.4.120 | ✅ |
| opencv-python | (unpinned) | 5.0.0 | ✅ |
| numpy | (unpinned) | 2.5.2 | ✅ |
| pyyaml | (unpinned) | via `yaml` | ✅ |
| streamlit | (unpinned) | 1.61.1 | ✅ |
| paramiko | `>=3.5.1` | 5.0.0 | ✅ |
| torch | **missing** | 2.13.0 | ⚠️ see note |
| torchvision | **missing** | 0.28.0 | ⚠️ see note |

> **⚠️ torch / torchvision** are not declared in `requirements.txt` but are
> installed transitively by `ultralytics`.  They should be pinned explicitly
> for reproducible environments and to select the correct variant
> (CPU-only vs CUDA vs ARM64).

---

## Optional Dependencies

| Package | Purpose | Tested Version | Notes |
|---|---|---|---|
| onnx | ONNX graph validation | 1.22.0 | Recommended |
| onnxruntime | ONNX inference / validation | 1.28.0 | Recommended |
| hailo_sdk_client | ONNX → HEF conversion | N/A | Proprietary — see below |

### Hailo SDK

The `hailo_sdk_client` package is **proprietary** and must be installed
separately:

1. Register at <https://developer.hailo.ai>
2. Download the SDK matching your target hardware (`hailo8`, `hailo8l`, or `hailo8r`)
3. Install with `pip install hailo_sdk_client-<version>.whl`

The code gracefully handles its absence with an `ImportError` pointing to the
download page.

---

## Known Working Version Set (2026-08)

```
# requirements-pinned.txt  (reference, not enforced)
ultralytics==8.4.120
opencv-python==5.0.0
numpy==2.5.2
pyyaml>=6.0
streamlit==1.61.1
paramiko==5.0.0

# Transitive (install explicitly for reproducibility)
torch==2.13.0
torchvision==0.28.0

# Optional / export pipeline
onnx==1.22.0
onnxruntime==1.28.0
```

---

## Architecture Notes

### x86-64 (development / CI)

All packages install from PyPI wheels without additional steps.

### ARM64 / aarch64 (Raspberry Pi 5)

- `torch` and `torchvision` may require alternative wheels.
  Check <https://pytorch.org/get-started/locally/> for the ARM64 builds.
- `opencv-python` has ARM64 wheels on PyPI since version 4.8.
- Hailo SDK provides a separate `.whl` for ARM64 via the Developer Zone.

---

## Breaking Changes Between Versions

### ultralytics

| Change | Version | Impact |
|---|---|---|
| `model.train()` `data=` kwarg renamed | 8.0 → 8.1 | Low — named arg unchanged |
| ONNX export `opset` default changed from 11 to 12 | 8.2 | Low — pinned to 12 in export_hailo.py |
| `model.export(simplify=True)` requires `onnxsim` | 8.0+ | Low — optional |

### streamlit

| Change | Version | Impact |
|---|---|---|
| `st.file_uploader(type=["zip"])` | All 1.x | No breaking change — API stable |
| `st.experimental_*` APIs removed | 1.38+ | Medium — not used in this project |
| `st.set_page_config` must be first call | 1.28+ | Low — already respected |

### paramiko

| Change | Version | Impact |
|---|---|---|
| `RejectPolicy` added | 3.0 | None — used in onnx_to_hef.py |
| `AutoAddPolicy` security deprecation warning | 3.2+ | Low — documented in code |

---

## Installation Warnings / Errors

| Symptom | Likely Cause | Resolution |
|---|---|---|
| `ModuleNotFoundError: hailo_sdk_client` | Proprietary SDK not installed | Install from developer.hailo.ai |
| `torch` not found | Not declared in requirements.txt | `pip install torch torchvision` |
| `onnx` not found | Not in requirements.txt | `pip install onnx onnxruntime` |
| `No module named 'cv2'` | `opencv-python` install failed | `pip install opencv-python` |
| ARM64 torch wheel error | Wrong torch index URL | Use `--index-url` for PyTorch ARM |

---

## Test Suite

The dependency test suite lives in `tests/test_dependencies.py` and covers:

- Python version requirement (≥3.10 per pyproject.toml, ≥3.9 for Pi compatibility)
- ARM64 architecture detection
- All required packages in requirements.txt (import + minimum version)
- Optional packages (onnx, onnxruntime, hailo_sdk_client) — non-fatal
- `torch.cuda.is_available()` reporting
- Ultralytics YOLO API surface (`YOLO`, `train`, `export`, `predict`, `val`)
- Streamlit API surface (`file_uploader`, `selectbox`, `sidebar`)
- ONNX pipeline imports (`onnx.checker`, `onnxruntime.InferenceSession`)
- All script entry points (`scripts/*.py`) importable without side-effects
- Dashboard utilities (`dashboard/path_utils.py`) importable

Run with:

```bash
pytest tests/test_dependencies.py -v
```
