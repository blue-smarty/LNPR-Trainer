#!/usr/bin/env python3
"""Streamlit dashboard for LNPR Trainer workflows."""

from __future__ import annotations

from pathlib import Path
import sys
import subprocess
import json

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.validation import (
    validate_setup_params,
    validate_train_params,
    validate_export_params,
    validate_hef_params,
)
from dashboard.artifacts import (
    find_recent_runs,
    find_all_onnx,
    find_all_hef,
    format_size,
    format_mtime,
    infer_onnx_path,
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

                    data_yaml_path = setup_dataset(
                        root_path=dataset_root, classes_csv=classes
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

    data_yaml = st.selectbox(
        "Path to data.yaml",
        options=list_paths("**/data.yaml", "data/lnpr_dataset/data.yaml"),
        help="Select the data.yaml that describes your LNPR dataset.",
    )

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
        result = validate_train_params(
            data_yaml=data_yaml,
            model_name=model_name,
            epochs=int(epochs),
            imgsz=int(imgsz),
            batch=int(batch),
            project=project,
            repo_root=REPO_ROOT,
        )
        if show_validation(result):
            data_path = (REPO_ROOT / data_yaml).resolve()
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

                    train_kwargs = {
                        "data": str(data_path),
                        "epochs": int(epochs),
                        "imgsz": int(imgsz),
                        "batch": int(batch),
                        "project": project,
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
                except ImportError as exc:
                    st.error(str(exc))
                    st.info(
                        "Install the Hailo SDK from https://developer.hailo.ai "
                        "and re-run the conversion."
                    )
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
        "Use cropped plate images with `train/images`, `val/images`, and `labels.txt` files."
    )

    rec_action = st.radio(
        "Recognition action",
        ["Train recognizer", "Export recognizer ONNX", "Run end-to-end inference"],
        horizontal=True,
    )

    if rec_action == "Train recognizer":
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
            try:
                cmd = [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "train_recognizer.py"),
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
                if exc.stdout:
                    st.code(exc.stdout, language="text")
                if exc.stderr:
                    st.code(exc.stderr, language="text")
                show_exception(exc)
            except Exception as exc:
                show_exception(exc)

    elif rec_action == "Export recognizer ONNX":
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
            try:
                cmd = [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "export_recognizer_onnx.py"),
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
                if exc.stdout:
                    st.code(exc.stdout, language="text")
                if exc.stderr:
                    st.code(exc.stderr, language="text")
                show_exception(exc)
            except Exception as exc:
                show_exception(exc)

    else:
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
            try:
                cmd = [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "infer_plate_text.py"),
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
                if exc.stdout:
                    st.code(exc.stdout, language="text")
                if exc.stderr:
                    st.code(exc.stderr, language="text")
                show_exception(exc)
            except Exception as exc:
                show_exception(exc)
