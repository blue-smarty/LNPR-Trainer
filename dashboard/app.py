#!/usr/bin/env python3
"""Streamlit dashboard for LNPR Trainer workflows."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import subprocess
import json
import tempfile
import zipfile

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.validation import (
    validate_dataset_source,
    validate_setup_params,
    validate_train_params,
    validate_export_params,
    validate_hef_params,
    validate_deploy_params,
)
from dashboard.artifacts import (
    find_recent_runs,
    find_all_onnx,
    find_all_hef,
    format_size,
    format_mtime,
    infer_onnx_path,
)
from dashboard.path_utils import (
    resolve_dataset_path_or_url as _resolve_dataset_path_or_url,
    resolve_local_path as _resolve_local_path,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COMMON_MODELS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
    "custom / enter below",
]

RECOGNITION_ACTION_SCRIPTS = {
    "Train recognizer": "train_recognizer.py",
    "Export recognizer ONNX": "export_recognizer_onnx.py",
    "Run end-to-end inference": "infer_plate_text.py",
}


def list_paths(pattern: str, default_value: str) -> list[str]:
    options: set[str] = {default_value}
    for path in REPO_ROOT.glob(pattern):
        if path.is_file():
            options.add(str(path.relative_to(REPO_ROOT)))
    return sorted(options)


def show_validation(result) -> bool:
    """Render validation errors and warnings; return True when safe to proceed."""
    for msg in result.warnings:
        st.warning(msg)
    for msg in result.errors:
        st.error(msg)
    return result.ok


def show_exception(exc: Exception) -> None:
    st.error(f"Operation failed: {exc}")
    with st.expander("Show traceback"):
        import traceback
        st.code(traceback.format_exc(), language="text")


def run_subprocess(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)


def get_script_path(script_name: str) -> Path | None:
    """Return script path when present."""
    script_path = REPO_ROOT / "scripts" / script_name
    return script_path if script_path.exists() else None


def ensure_script_exists(script_name: str) -> Path | None:
    """Return script path when present; otherwise show a dashboard error."""
    script_path = get_script_path(script_name)
    if script_path is not None:
        return script_path
    script_path = REPO_ROOT / "scripts" / script_name
    st.error(f"Required script not found: {script_path}")
    st.info(
        "This recognition action is unavailable because the required script is "
        "missing in your checkout."
    )
    return None


def _handle_uploaded_dataset(uploaded_file, tmp_dir: Path) -> Path:
    """Save an uploaded ZIP archive to *tmp_dir*, extract it, and return the
    path to the ``data.yaml`` found inside.

    Raises ``FileNotFoundError`` when no ``data.yaml`` is present in the archive.
    Raises ``zipfile.BadZipFile`` when the uploaded file is not a valid ZIP.
    """
    zip_path = tmp_dir / uploaded_file.name
    zip_path.write_bytes(uploaded_file.getvalue())

    extract_dir = tmp_dir / "dataset_upload"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Reject archive members that would escape the extraction directory.
        extract_root = extract_dir.resolve()
        for member in zf.infolist():
            member_path = (extract_dir / member.filename).resolve()
            if not str(member_path).startswith(str(extract_root)):
                raise ValueError(f"Unsafe path in archive member: {member.filename}")
        zf.extractall(extract_dir)

    yaml_files = sorted(extract_dir.rglob("data.yaml"))
    if not yaml_files:
        raise FileNotFoundError(
            "No data.yaml found in the uploaded archive. "
            "Ensure the ZIP contains a valid YOLO dataset with a data.yaml file."
        )
    return yaml_files[0]


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="LNPR Trainer Dashboard", layout="wide")
st.title("LNPR Trainer Dashboard")
st.caption(
    "Train and export License Plate Number Recognition (LNPR) models — "
    "dataset setup, YOLOv8 training, ONNX export, Hailo HEF conversion, and plate recognition in one place."
)

# ---------------------------------------------------------------------------
# Sidebar — recent artifacts summary
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Recent Artifacts")
    _runs_root = REPO_ROOT / "runs" / "detect"
    _recent_runs = find_recent_runs(_runs_root, max_runs=5)
    if _recent_runs:
        for run in _recent_runs:
            with st.expander(f"📁 {run.name}", expanded=False):
                st.caption(f"Modified: {format_mtime(run.path)}")
                if run.best_pt:
                    st.markdown(f"✅ `best.pt` ({format_size(run.best_pt)})")
                if run.last_pt:
                    st.markdown(f"📄 `last.pt` ({format_size(run.last_pt)})")
                for onnx in run.onnx_files:
                    st.markdown(f"🔷 `{onnx.name}` ({format_size(onnx)})")
                for hef in run.hef_files:
                    st.markdown(f"🟢 `{hef.name}` ({format_size(hef)})")
    else:
        st.info("No training runs found yet.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_setup, tab_train, tab_export, tab_hef, tab_artifacts, tab_recognition = st.tabs(
    ["Setup Dataset", "Train Model", "Export ONNX", "Convert to HEF", "Artifacts", "Recognition"]
)

# ── Setup Dataset ────────────────────────────────────────────────────────────

with tab_setup:
    st.subheader("Create LNPR dataset structure")
    st.markdown(
        "Creates the standard YOLOv8 directory layout and writes a `data.yaml` "
        "for your license plate dataset. Use `license_plate` as the default class, "
        "or add additional classes (e.g. `license_plate,motorcycle_plate`) if needed."
    )

    dataset_root = st.text_input(
        "Dataset root",
        value="data/lnpr_dataset",
        help="Path (relative to repo root or absolute) where the dataset will be created.",
    )
    classes = st.text_input(
        "Classes (comma-separated)",
        value="license_plate",
        help="One or more class names separated by commas. Default is `license_plate`.",
    )

    if st.button("Run dataset setup", type="primary"):
        result = validate_setup_params(dataset_root, classes)
        if show_validation(result):
            with st.spinner("Creating dataset structure…"):
                try:
                    from scripts.setup_dataset import setup_dataset

                    # Resolve the root path against REPO_ROOT so that relative
                    # paths are anchored to the repository root regardless of
                    # the process working directory.
                    data_yaml_path = setup_dataset(
                        root_path=str(_resolve_local_path(dataset_root, REPO_ROOT)),
                        classes_csv=classes,
                    )
                    st.success("Dataset structure created successfully.")
                    st.markdown("**Generated file**")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.code(str(data_yaml_path), language="text")
                    with col2:
                        st.metric("data.yaml", format_size(data_yaml_path))

                    with st.expander("Preview data.yaml"):
                        with open(data_yaml_path, "r", encoding="utf-8") as fh:
                            st.code(fh.read(), language="yaml")
                except Exception as exc:  # pragma: no cover - UI feedback path
                    show_exception(exc)

# ── Train Model ──────────────────────────────────────────────────────────────

with tab_train:
    st.subheader("Train LNPR YOLOv8 model")

    st.markdown("**Dataset source**")
    st.caption(
        "Provide your training dataset using one of the options below. "
        "If multiple sources are filled in, **uploaded file** takes highest priority, "
        "then **path / URL**, then **existing data.yaml**."
    )

    uploaded_dataset = st.file_uploader(
        "Upload dataset archive (.zip)",
        type=["zip"],
        key="train_dataset_upload",
        help=(
            "Upload a ZIP archive that contains your YOLO dataset: "
            "`images/`, `labels/`, and a `data.yaml` file. "
            "This takes priority over all other dataset sources."
        ),
    )

    dataset_path_or_url = st.text_input(
        "Dataset path or URL (fallback)",
        value="",
        key="train_dataset_path_url",
        help=(
            "Enter a local path or an http(s) URL pointing to your dataset. "
            "Accepted formats: a `data.yaml` file, a directory containing `data.yaml`, "
            "or a `.zip` archive. Used only if no file is uploaded above."
        ),
    )

    data_yaml = st.selectbox(
        "Existing data.yaml (fallback)",
        options=list_paths("**/data.yaml", "data/lnpr_dataset/data.yaml"),
        key="train_data_yaml",
        help=(
            "Select a data.yaml already present in the repository. "
            "Used only if no file is uploaded and no path/URL is entered above."
        ),
    )

    st.divider()

    model_choice = st.selectbox(
        "Model",
        options=COMMON_MODELS,
        index=0,
        help=(
            "Choose a YOLOv8 model size. `yolov8n` is recommended for edge deployment "
            "on Raspberry Pi 5 + Hailo-8L."
        ),
    )
    if model_choice == "custom / enter below":
        model_name = st.text_input(
            "Custom model name or path",
            value="yolov8n.pt",
            help="Enter the model file name (downloaded automatically) or a path to a local .pt file.",
        )
    else:
        model_name = model_choice

    col_ep, col_img, col_bat = st.columns(3)
    with col_ep:
        epochs = st.number_input(
            "Epochs",
            min_value=1,
            value=50,
            help="Number of full passes through the training data.",
        )
    with col_img:
        imgsz = st.number_input(
            "Image size",
            min_value=32,
            value=640,
            step=32,
            help="Input resolution (pixels). Must be a multiple of 32.",
        )
    with col_bat:
        batch = st.number_input(
            "Batch size",
            min_value=1,
            value=16,
            help="Number of images processed per training step.",
        )

    with st.expander("Advanced options"):
        col_proj, col_name = st.columns(2)
        with col_proj:
            project = st.text_input(
                "Project directory",
                value="runs/detect",
                help="Parent folder where run output is saved.",
            )
        with col_name:
            run_name = st.text_input(
                "Run name",
                value="lnpr",
                help="Sub-folder name inside the project directory for this run.",
            )
        device = st.text_input(
            "Device",
            value="",
            help="Training device: leave blank for auto-detect, `cpu` for CPU, `0` for first GPU.",
        )
        cfg = st.text_input(
            "Config yaml (optional)",
            value="",
            help="Path to an Ultralytics trainer config file (e.g. `configs/yolov8-lnpr.yaml`).",
        )
        resume = st.checkbox(
            "Resume previous run",
            help="Continue training from the last saved checkpoint of a previous run.",
        )

    if st.button("Run training", type="primary"):
        # ── Validate dataset source ────────────────────────────────────────────
        src_result = validate_dataset_source(
            uploaded_file_name=uploaded_dataset.name if uploaded_dataset is not None else None,
            path_or_url=dataset_path_or_url,
            data_yaml=data_yaml,
            repo_root=REPO_ROOT,
        )
        if not show_validation(src_result):
            pass  # errors displayed above; stop here
        else:
            # ── Resolve data.yaml from whichever source was provided ───────────
            _tmp_dir: Path | None = None
            try:
                resolved_data_yaml: str
                if uploaded_dataset is not None:
                    _tmp_dir = Path(tempfile.mkdtemp(prefix="lnpr_dataset_"))
                    with st.spinner("Extracting uploaded dataset archive…"):
                        resolved_path = _handle_uploaded_dataset(
                            uploaded_dataset, _tmp_dir
                        )
                    st.info(
                        "Using uploaded archive — dataset loaded from "
                        f"`{resolved_path.name}`."
                    )
                    resolved_data_yaml = str(resolved_path)
                    # validate_train_params receives an absolute path; skip repo_root
                    effective_repo_root: Path | None = None
                elif dataset_path_or_url.strip():
                    _tmp_dir = Path(tempfile.mkdtemp(prefix="lnpr_dataset_"))
                    with st.spinner("Resolving dataset path / URL…"):
                        resolved_path = _resolve_dataset_path_or_url(
                            dataset_path_or_url, _tmp_dir
                        )
                    resolved_data_yaml = str(resolved_path)
                    effective_repo_root = None
                else:
                    # Resolve the selected data.yaml path against REPO_ROOT so
                    # the result is always absolute.  Using _resolve_local_path
                    # prevents CWD-dependent resolution that can produce
                    # duplicated path segments such as
                    # ``…/dashboard/dashboard/data/lnpr_dataset/data.yaml``.
                    resolved_data_yaml = str(_resolve_local_path(data_yaml, REPO_ROOT))
                    effective_repo_root = None  # path is now absolute

                result = validate_train_params(
                    data_yaml=resolved_data_yaml,
                    model_name=model_name,
                    epochs=int(epochs),
                    imgsz=int(imgsz),
                    batch=int(batch),
                    project=project,
                    repo_root=effective_repo_root,
                )
                if show_validation(result):
                    data_path = Path(resolved_data_yaml).resolve()
                    # Pre-train existence guard: emit a clear error if the
                    # resolved absolute path does not exist so the user sees a
                    # descriptive message instead of a cryptic Ultralytics
                    # RuntimeError.
                    if not data_path.exists():
                        st.error(
                            f"Dataset configuration file not found: `{data_path}`. "
                            "Verify the dataset path is correct and the file exists "
                            "before starting training."
                        )
                    else:
                        progress_text = st.empty()
                        progress_bar = st.progress(0, text="Preparing training…")
                        try:
                            from ultralytics import YOLO

                            progress_state = {"last_epoch": 0, "target_epochs": int(epochs)}

                            def on_train_start(trainer) -> None:
                                total_epochs = int(getattr(trainer, "epochs", progress_state["target_epochs"]))
                                progress_state["target_epochs"] = max(total_epochs, 1)
                                progress_text.info(f"Training started: 0/{progress_state['target_epochs']} epochs")
                                progress_bar.progress(0, text=f"Training 0/{progress_state['target_epochs']} epochs")

                            def on_train_epoch_end(trainer) -> None:
                                current_epoch = int(getattr(trainer, "epoch", -1)) + 1
                                total_epochs = int(getattr(trainer, "epochs", progress_state["target_epochs"]))
                                total_epochs = max(total_epochs, 1)
                                progress_state["last_epoch"] = current_epoch
                                fraction = min(current_epoch / total_epochs, 1.0)
                                progress_bar.progress(fraction, text=f"Training {current_epoch}/{total_epochs} epochs")
                                progress_text.info(f"Training progress: {current_epoch}/{total_epochs} epochs")

                            with st.spinner("Training in progress — this may take a while…"):
                                model = YOLO(model_name)
                                model.add_callback("on_train_start", on_train_start)
                                model.add_callback("on_train_epoch_end", on_train_epoch_end)

                                project_path = str(REPO_ROOT / project)
                                train_kwargs = {
                                    "data": str(data_path),
                                    "epochs": int(epochs),
                                    "imgsz": int(imgsz),
                                    "batch": int(batch),
                                    "project": project_path,
                                    "name": run_name,
                                    "resume": resume,
                                }

                                if device.strip():
                                    train_kwargs["device"] = device.strip()

                                if cfg.strip():
                                    train_kwargs["cfg"] = cfg.strip()

                                model.train(**train_kwargs)

                            progress_bar.progress(1.0, text=f"Training complete: {progress_state['target_epochs']}/{progress_state['target_epochs']} epochs")
                            progress_text.success("Training completed successfully.")

                            run_dir = REPO_ROOT / project / run_name
                            weights_dir = run_dir / "weights"
                            st.markdown("**Training output**")
                            st.code(str(run_dir), language="text")

                            if weights_dir.exists():
                                found_weights = sorted(weights_dir.glob("*.pt"))
                                if found_weights:
                                    st.markdown("**Weights found:**")
                                    for wt in found_weights:
                                        st.markdown(f"- `{wt.relative_to(REPO_ROOT)}` ({format_size(wt)})")
                            else:
                                st.info("Weights directory not found yet; check the run directory above.")
                        except Exception as exc:  # pragma: no cover - UI feedback path
                            progress_bar.empty()
                            progress_text.empty()
                            show_exception(exc)
            except Exception as exc:  # dataset resolution / extraction errors
                show_exception(exc)
            finally:
                if _tmp_dir is not None:
                    shutil.rmtree(_tmp_dir, ignore_errors=True)

# ── Export ONNX ───────────────────────────────────────────────────────────────

with tab_export:
    st.subheader("Export trained LNPR model to ONNX")
    st.markdown(
        "Export a trained `.pt` weights file to ONNX format, ready for the "
        "Hailo Dataflow Compiler."
    )

    weights = st.selectbox(
        "Weights path",
        options=list_paths("**/*.pt", "runs/detect/lnpr/weights/best.pt"),
        key="weights",
        help="Select a trained .pt weights file to export.",
    )

    col_ei, col_eb = st.columns(2)
    with col_ei:
        export_imgsz = st.number_input(
            "Image size",
            min_value=32,
            value=640,
            step=32,
            key="ex_img",
            help="Input resolution to bake into the ONNX graph.",
        )
    with col_eb:
        export_batch = st.number_input(
            "Batch size",
            min_value=1,
            value=1,
            key="ex_batch",
            help="Batch dimension in the ONNX model (usually 1 for Hailo).",
        )

    with st.expander("Advanced options"):
        opset = st.number_input(
            "ONNX opset",
            min_value=9,
            value=12,
            help="ONNX operator-set version. Hailo recommends opset 11 or 12.",
        )
        dynamic = st.checkbox(
            "Dynamic shapes",
            help="Enable variable batch/spatial dimensions in the exported ONNX.",
        )

    if st.button("Run ONNX export", type="primary"):
        result = validate_export_params(
            weights=weights,
            imgsz=int(export_imgsz),
            batch=int(export_batch),
            opset=int(opset),
            repo_root=REPO_ROOT,
        )
        if show_validation(result):
            weights_path = (REPO_ROOT / weights).resolve()
            with st.spinner("Exporting to ONNX…"):
                try:
                    from scripts.export_hailo import export_onnx

                    export_onnx(
                        weights=str(weights_path),
                        imgsz=int(export_imgsz),
                        batch=int(export_batch),
                        opset=int(opset),
                        dynamic=dynamic,
                    )
                    st.success("ONNX export completed.")

                    onnx_path = infer_onnx_path(weights_path)
                    if onnx_path:
                        st.markdown("**Exported file**")
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.code(str(onnx_path), language="text")
                        with col2:
                            st.metric("ONNX size", format_size(onnx_path))
                    else:
                        st.info(
                            "ONNX file not found next to weights. "
                            "Check the weights directory for a .onnx file."
                        )
                except Exception as exc:  # pragma: no cover - UI feedback path
                    show_exception(exc)

# ── Convert to HEF ────────────────────────────────────────────────────────────

with tab_hef:
    st.subheader("Convert LNPR ONNX model to Hailo HEF")
    st.markdown(
        "Compile a `.onnx` LNPR model to a `.hef` file using the **Hailo Dataflow Compiler**. "
        "Requires the [Hailo SDK](https://developer.hailo.ai) to be installed."
    )

    all_onnx_files = find_all_onnx(REPO_ROOT)
    onnx_options = (
        [str(p.relative_to(REPO_ROOT)) for p in all_onnx_files]
        if all_onnx_files
        else ["(no .onnx files found — run Export ONNX first)"]
    )
    onnx_input = st.selectbox(
        "ONNX file",
        options=onnx_options,
        help="Select the .onnx file to compile.",
    )

    hw_arch = st.selectbox(
        "Hardware architecture",
        options=["hailo8l", "hailo8", "hailo8r"],
        index=0,
        help=(
            "`hailo8l` — Hailo-8L (Raspberry Pi 5 AI HAT+, 13 TOPS)  \n"
            "`hailo8` — Hailo-8 (26 TOPS)  \n"
            "`hailo8r` — Hailo-8R"
        ),
    )

    calib_path = st.text_input(
        "Calibration images directory (optional)",
        value="",
        help=(
            "Path to a folder of representative license plate images. "
            "Providing real images produces more accurate INT8 quantization. "
            "Leave blank to use random calibration data as a fallback."
        ),
    )

    with st.expander("Advanced options"):
        col_imgsz, col_outdir = st.columns(2)
        with col_imgsz:
            hef_imgsz = st.number_input(
                "Image size",
                min_value=32,
                value=640,
                step=32,
                key="hef_img",
                help="Model input resolution (must match the value used during ONNX export).",
            )
        with col_outdir:
            output_dir = st.text_input(
                "Output directory (optional)",
                value="",
                key="hef_outdir",
                help="Where to save the .hef file. Defaults to the same directory as the ONNX file.",
            )

    if st.button("Run HEF conversion", type="primary"):
        result = validate_hef_params(
            onnx_path=onnx_input,
            hw_arch=hw_arch,
            calib_path=calib_path,
            repo_root=REPO_ROOT,
        )
        if show_validation(result):
            onnx_file_path = (REPO_ROOT / onnx_input).resolve()
            out_dir = (REPO_ROOT / output_dir).resolve() if output_dir.strip() else None
            with st.spinner("Compiling ONNX → HEF — this may take several minutes…"):
                try:
                    from scripts.onnx_to_hef import convert_onnx_to_hef

                    hef_path = convert_onnx_to_hef(
                        onnx_path=str(onnx_file_path),
                        output_dir=str(out_dir) if out_dir else None,
                        hw_arch=hw_arch,
                        calib_path=calib_path.strip() if calib_path.strip() else None,
                        input_shape=(int(hef_imgsz), int(hef_imgsz)),
                    )
                    st.success("HEF compilation completed.")
                    st.markdown("**Output file**")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.code(str(hef_path), language="text")
                    with col2:
                        st.metric("HEF size", format_size(hef_path))
                    st.session_state["last_hef_path"] = str(hef_path)
                except ImportError as exc:
                    st.error(str(exc))
                    st.info(
                        "Install the Hailo SDK from https://developer.hailo.ai "
                        "and re-run the conversion."
                    )
                except Exception as exc:  # pragma: no cover - UI feedback path
                    show_exception(exc)

    st.divider()
    st.markdown("#### Deploy HEF to Hailo device")
    st.markdown(
        "Copy the converted `.hef` file directly to a Hailo device (e.g. Raspberry Pi + "
        "Hailo-8L HAT) over SSH/SFTP."
    )

    deploy_host = st.text_input(
        "Device host / IP",
        value="",
        key="deploy_host",
        help="Hostname or IP address of the target Hailo device.",
    )
    deploy_user = st.text_input(
        "SSH username",
        value="pi",
        key="deploy_user",
        help="SSH username on the remote device.",
    )
    deploy_remote_path = st.text_input(
        "Remote destination path",
        value="/home/pi/models/",
        key="deploy_remote_path",
        help=(
            "Absolute path on the device where the HEF will be stored. "
            "Paths ending with '/' are treated as directories."
        ),
    )

    with st.expander("SSH authentication"):
        deploy_password = st.text_input(
            "SSH password (optional)",
            value="",
            type="password",
            key="deploy_password",
            help="Leave blank to use an SSH key instead.",
        )
        deploy_key_path = st.text_input(
            "SSH private key path (optional)",
            value="",
            key="deploy_key_path",
            help="Path to your local private key, e.g. ~/.ssh/id_rsa.",
        )
        deploy_port = st.number_input(
            "SSH port",
            min_value=1,
            max_value=65535,
            value=22,
            key="deploy_port",
            help="SSH port on the remote device (default: 22).",
        )
        deploy_timeout = st.number_input(
            "Connection timeout (seconds)",
            min_value=1,
            max_value=300,
            value=30,
            key="deploy_timeout",
            help="Seconds to wait for the SSH connection before giving up (default: 30).",
        )
        deploy_accept_unknown = st.checkbox(
            "Accept unknown host key",
            key="deploy_accept_unknown",
            help=(
                "Automatically accept the device's SSH host key if not already in "
                "~/.ssh/known_hosts. Only enable on trusted private networks."
            ),
        )

    deploy_hef_input = st.text_input(
        "HEF file to deploy",
        value=st.session_state.get("last_hef_path", ""),
        key="deploy_hef_input",
        help=(
            "Path to the .hef file to upload. "
            "Auto-filled after a successful conversion above."
        ),
    )

    if st.button("Deploy HEF to device", type="secondary"):
        deploy_result = validate_deploy_params(
            host=deploy_host,
            username=deploy_user,
            remote_path=deploy_remote_path,
            password=deploy_password,
            key_path=deploy_key_path,
        )
        hef_deploy_path = deploy_hef_input.strip()
        if not hef_deploy_path:
            st.error("HEF file path must not be empty. Run the conversion above first.")
        elif not Path(hef_deploy_path).exists():
            st.error(f"HEF file not found: {hef_deploy_path}")
        elif show_validation(deploy_result):
            with st.spinner(f"Deploying {Path(hef_deploy_path).name} to {deploy_host}…"):
                try:
                    from scripts.onnx_to_hef import deploy_hef_to_device

                    remote_dest = deploy_hef_to_device(
                        hef_path=hef_deploy_path,
                        host=deploy_host.strip(),
                        username=deploy_user.strip(),
                        remote_path=deploy_remote_path.strip(),
                        password=deploy_password.strip() if deploy_password.strip() else None,
                        key_path=deploy_key_path.strip() if deploy_key_path.strip() else None,
                        port=int(deploy_port),
                        accept_unknown_host_key=deploy_accept_unknown,
                        timeout=float(deploy_timeout),
                    )
                    st.success(
                        f"HEF deployed successfully to `{deploy_host}:{remote_dest}`"
                    )
                except ImportError as exc:
                    st.error(str(exc))
                    st.info("Install paramiko with: pip install paramiko")
                except Exception as exc:  # pragma: no cover - UI feedback path
                    show_exception(exc)

# ── Artifacts ────────────────────────────────────────────────────────────────

with tab_artifacts:
    st.subheader("Artifacts Browser")
    st.markdown(
        "Browse recent LNPR training runs, weights, and exported ONNX/HEF files without "
        "leaving the dashboard."
    )

    runs_root = REPO_ROOT / "runs" / "detect"
    recent_runs = find_recent_runs(runs_root, max_runs=20)

    if not recent_runs:
        st.info(
            f"No training runs found under `{runs_root.relative_to(REPO_ROOT)}`. "
            "Complete a training run first."
        )
    else:
        st.markdown(f"**{len(recent_runs)} run(s) found** — sorted newest first")
        for run in recent_runs:
            with st.expander(f"📁 {run.name}  —  {format_mtime(run.path)}", expanded=False):
                st.markdown(f"**Path:** `{run.path}`")

                if run.weights:
                    st.markdown("**Weights:**")
                    for wt in run.weights:
                        label = "✅ best.pt" if wt.name == "best.pt" else f"📄 {wt.name}"
                        st.markdown(
                            f"- {label} &nbsp; `{wt}` &nbsp; ({format_size(wt)}, "
                            f"modified {format_mtime(wt)})"
                        )
                else:
                    st.caption("No .pt weights found in this run.")

                if run.onnx_files:
                    st.markdown("**ONNX exports:**")
                    for onnx in run.onnx_files:
                        st.markdown(
                            f"- 🔷 `{onnx}` ({format_size(onnx)}, "
                            f"modified {format_mtime(onnx)})"
                        )

                if run.hef_files:
                    st.markdown("**HEF files:**")
                    for hef in run.hef_files:
                        st.markdown(
                            f"- 🟢 `{hef}` ({format_size(hef)}, "
                            f"modified {format_mtime(hef)})"
                        )

    st.divider()
    st.markdown("#### All ONNX files in repository")
    all_onnx = find_all_onnx(REPO_ROOT)
    if all_onnx:
        for onnx in all_onnx:
            st.markdown(
                f"- 🔷 `{onnx}` ({format_size(onnx)}, modified {format_mtime(onnx)})"
            )
    else:
        st.info("No .onnx files found in the repository yet.")

    st.divider()
    st.markdown("#### All HEF files in repository")
    all_hef = find_all_hef(REPO_ROOT)
    if all_hef:
        for hef in all_hef:
            st.markdown(
                f"- 🟢 `{hef}` ({format_size(hef)}, modified {format_mtime(hef)})"
            )
    else:
        st.info("No .hef files found in the repository yet.")

# ── Recognition ──────────────────────────────────────────────────────────────

with tab_recognition:
    st.subheader("Plate text recognition")
    st.markdown(
        "Train, export, and run inference for the CRNN-based plate recognizer. "
        "Use cropped plate images with `train/images`, `val/images`, and split label files "
        "such as `train/labels.txt` or `labels/train.txt`."
    )

    # ── Upload Dataset Files ──────────────────────────────────────────────────
    with st.expander("📂 Upload dataset files", expanded=False):
        st.markdown(
            "Upload cropped plate images and/or a labels text file into the recognition "
            "dataset directory structure. Images are saved to `{root}/{split}/images/` and "
            "the labels file is saved to `{root}/labels/{split}.txt`."
        )
        upload_root = st.text_input(
            "Dataset root", value="data/recognition", key="upload_rec_root"
        )
        upload_split = st.selectbox(
            "Split", ["train", "val"], key="upload_rec_split"
        )

        upload_images = st.file_uploader(
            "Image files (.jpg / .jpeg / .png)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="upload_rec_images",
        )
        upload_labels = st.file_uploader(
            "Labels text file (.txt)  —  one `<image_filename> <text>` entry per line",
            type=["txt"],
            accept_multiple_files=False,
            key="upload_rec_labels",
        )

        images_dest = REPO_ROOT / upload_root / upload_split / "images"
        labels_dest = REPO_ROOT / upload_root / "labels" / f"{upload_split}.txt"
        st.caption(f"Images will be saved to: `{images_dest.relative_to(REPO_ROOT)}`")
        st.caption(f"Labels file will be saved to: `{labels_dest.relative_to(REPO_ROOT)}`")

        if st.button("Upload files", key="upload_rec_btn"):
            if not upload_images and not upload_labels:
                st.warning("No files selected. Please choose at least one image or a labels file.")
            else:
                upload_errors: list[str] = []
                saved_images: list[str] = []

                if upload_images:
                    try:
                        images_dest.mkdir(parents=True, exist_ok=True)
                        for uf in upload_images:
                            dest_file = images_dest / uf.name
                            dest_file.write_bytes(uf.read())
                            saved_images.append(uf.name)
                    except Exception as exc:
                        upload_errors.append(f"Failed to save images: {exc}")

                if upload_labels is not None:
                    try:
                        labels_dest.parent.mkdir(parents=True, exist_ok=True)
                        labels_dest.write_bytes(upload_labels.read())
                    except Exception as exc:
                        upload_errors.append(f"Failed to save labels file: {exc}")

                if upload_errors:
                    for err in upload_errors:
                        st.error(err)
                else:
                    if saved_images:
                        st.success(
                            f"Saved {len(saved_images)} image(s) to "
                            f"`{images_dest.relative_to(REPO_ROOT)}`"
                        )
                    if upload_labels is not None:
                        st.success(
                            f"Saved labels file to `{labels_dest.relative_to(REPO_ROOT)}`"
                        )

    st.divider()

    recognition_script_paths = {
        action: get_script_path(script_name)
        for action, script_name in RECOGNITION_ACTION_SCRIPTS.items()
    }
    available_recognition_actions = [
        action for action, script_path in recognition_script_paths.items() if script_path is not None
    ]
    missing_recognition_scripts = [
        script_name
        for action, script_name in RECOGNITION_ACTION_SCRIPTS.items()
        if recognition_script_paths[action] is None
    ]

    if not available_recognition_actions:
        st.warning("Recognition tools are not available in this checkout yet.")
        st.caption(
            "Missing scripts: "
            + ", ".join(f"`scripts/{script_name}`" for script_name in missing_recognition_scripts)
        )
    else:
        if missing_recognition_scripts:
            st.info(
                "Some recognition actions are hidden because their supporting scripts are missing: "
                + ", ".join(f"`scripts/{script_name}`" for script_name in missing_recognition_scripts)
            )

        rec_action = st.radio(
            "Recognition action",
            available_recognition_actions,
            horizontal=True,
        )

    if available_recognition_actions and rec_action == "Train recognizer":
        rec_data = st.text_input("Recognition dataset root", value="data/recognition")
        col1, col2, col3 = st.columns(3)
        with col1:
            rec_epochs = st.number_input("Epochs", min_value=1, value=30, key="rec_epochs")
        with col2:
            rec_batch = st.number_input("Batch size", min_value=1, value=32, key="rec_batch")
        with col3:
            rec_lr = st.number_input("Learning rate", min_value=0.000001, value=0.001, format="%.6f", key="rec_lr")

        col4, col5, col6 = st.columns(3)
        with col4:
            rec_h = st.number_input("Image height", min_value=16, value=32, key="rec_h")
        with col5:
            rec_w = st.number_input("Image width", min_value=32, value=160, key="rec_w")
        with col6:
            rec_workers = st.number_input("Workers", min_value=0, value=4, key="rec_workers")

        rec_charset = st.text_input("Charset", value="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        rec_hidden = st.number_input("RNN hidden size", min_value=32, value=256, step=32)
        rec_project = st.text_input("Project directory", value="runs/recognition", key="rec_project")
        rec_name = st.text_input("Run name", value="crnn_lp", key="rec_name")
        rec_device = st.text_input("Device", value="", key="rec_device")

        if st.button("Run recognizer training", type="primary"):
            script_path = ensure_script_exists("train_recognizer.py")
            if script_path is not None:
                try:
                    cmd = [
                        sys.executable,
                        str(script_path),
                        "--data",
                        rec_data,
                        "--epochs",
                        str(int(rec_epochs)),
                        "--batch",
                        str(int(rec_batch)),
                        "--img-height",
                        str(int(rec_h)),
                        "--img-width",
                        str(int(rec_w)),
                        "--lr",
                        str(float(rec_lr)),
                        "--workers",
                        str(int(rec_workers)),
                        "--charset",
                        rec_charset,
                        "--rnn-hidden-size",
                        str(int(rec_hidden)),
                        "--project",
                        rec_project,
                        "--name",
                        rec_name,
                    ]
                    if rec_device.strip():
                        cmd += ["--device", rec_device.strip()]

                    proc = run_subprocess(cmd)
                    st.success("Recognizer training completed successfully.")
                    if proc.stdout.strip():
                        st.code(proc.stdout, language="text")
                    if proc.stderr.strip():
                        st.code(proc.stderr, language="text")
                except subprocess.CalledProcessError as exc:
                    if exc.returncode == 2:
                        st.warning(
                            "Recognizer script exited with code 2, which usually means "
                            "invalid or unsupported command-line arguments."
                        )
                    if exc.stdout:
                        st.code(exc.stdout, language="text")
                    if exc.stderr:
                        st.code(exc.stderr, language="text")
                    show_exception(exc)
                except Exception as exc:
                    show_exception(exc)

    elif available_recognition_actions and rec_action == "Export recognizer ONNX":
        rec_weights = st.text_input(
            "Recognizer checkpoint",
            value="runs/recognition/crnn_lp/weights/best.pt",
        )
        col1, col2 = st.columns(2)
        with col1:
            rec_exp_h = st.number_input("Image height", min_value=16, value=32, key="rec_exp_h")
        with col2:
            rec_exp_w = st.number_input("Image width", min_value=32, value=160, key="rec_exp_w")
        rec_output = st.text_input("Output ONNX path (optional)", value="")
        rec_opset = st.number_input("ONNX opset", min_value=9, value=12, key="rec_opset")
        rec_dynamic = st.checkbox("Dynamic width", key="rec_dynamic")
        rec_export_device = st.text_input("Device", value="cpu", key="rec_export_device")

        if st.button("Export recognizer ONNX", type="primary"):
            script_path = ensure_script_exists("export_recognizer_onnx.py")
            if script_path is not None:
                try:
                    cmd = [
                        sys.executable,
                        str(script_path),
                        "--weights",
                        rec_weights,
                        "--img-height",
                        str(int(rec_exp_h)),
                        "--img-width",
                        str(int(rec_exp_w)),
                        "--opset",
                        str(int(rec_opset)),
                        "--device",
                        rec_export_device,
                    ]
                    if rec_output.strip():
                        cmd += ["--output", rec_output.strip()]
                    if rec_dynamic:
                        cmd.append("--dynamic-width")

                    proc = run_subprocess(cmd)
                    st.success("Recognizer ONNX export completed successfully.")
                    if proc.stdout.strip():
                        st.code(proc.stdout, language="text")
                    if proc.stderr.strip():
                        st.code(proc.stderr, language="text")
                except subprocess.CalledProcessError as exc:
                    if exc.returncode == 2:
                        st.warning(
                            "Recognizer export script exited with code 2, which usually "
                            "means invalid or unsupported command-line arguments."
                        )
                    if exc.stdout:
                        st.code(exc.stdout, language="text")
                    if exc.stderr:
                        st.code(exc.stderr, language="text")
                    show_exception(exc)
                except Exception as exc:
                    show_exception(exc)

    elif available_recognition_actions:
        infer_image = st.text_input("Image path", value="")
        infer_detector = st.text_input("Detector weights", value="runs/detect/lnpr/weights/best.pt")
        infer_recognizer = st.text_input("Recognizer checkpoint", value="runs/recognition/crnn_lp/weights/best.pt")

        col1, col2, col3 = st.columns(3)
        with col1:
            infer_imgsz = st.number_input("Detector image size", min_value=32, value=640, step=32)
        with col2:
            infer_conf = st.number_input("Confidence threshold", min_value=0.0, max_value=1.0, value=0.25)
        with col3:
            infer_iou = st.number_input("IoU threshold", min_value=0.0, max_value=1.0, value=0.45)

        col4, col5, col6 = st.columns(3)
        with col4:
            infer_max_det = st.number_input("Max detections", min_value=1, value=10)
        with col5:
            infer_rec_h = st.number_input("Recognizer height", min_value=16, value=32)
        with col6:
            infer_rec_w = st.number_input("Recognizer width", min_value=32, value=160)

        infer_device = st.text_input("Device", value="")
        infer_save_crops = st.checkbox("Save crops")
        infer_output_json = st.text_input("Output JSON path (optional)", value="")

        if st.button("Run end-to-end inference", type="primary"):
            script_path = ensure_script_exists("infer_plate_text.py")
            if script_path is not None:
                try:
                    cmd = [
                        sys.executable,
                        str(script_path),
                        "--image",
                        infer_image,
                        "--detector",
                        infer_detector,
                        "--recognizer",
                        infer_recognizer,
                        "--imgsz",
                        str(int(infer_imgsz)),
                        "--conf",
                        str(float(infer_conf)),
                        "--iou",
                        str(float(infer_iou)),
                        "--max-det",
                        str(int(infer_max_det)),
                        "--rec-img-height",
                        str(int(infer_rec_h)),
                        "--rec-img-width",
                        str(int(infer_rec_w)),
                    ]
                    if infer_device.strip():
                        cmd += ["--device", infer_device.strip()]
                    if infer_save_crops:
                        cmd.append("--save-crops")
                    if infer_output_json.strip():
                        cmd += ["--output-json", infer_output_json.strip()]

                    proc = run_subprocess(cmd)
                    st.success("Inference completed successfully.")
                    if proc.stdout.strip():
                        try:
                            parsed = json.loads(proc.stdout)
                            st.json(parsed)
                        except Exception:
                            st.code(proc.stdout, language="json")
                    if proc.stderr.strip():
                        st.code(proc.stderr, language="text")
                except subprocess.CalledProcessError as exc:
                    if exc.returncode == 2:
                        st.warning(
                            "Inference script exited with code 2, which usually means "
                            "invalid or unsupported command-line arguments."
                        )
                    if exc.stdout:
                        st.code(exc.stdout, language="text")
                    if exc.stderr:
                        st.code(exc.stderr, language="text")
                    show_exception(exc)
                except Exception as exc:
                    show_exception(exc)
