# LNPR-Trainer

A repository for training and exporting **License Plate Number Recognition (LNPR)** object detection models using YOLOv8/PyTorch, targeting Raspberry Pi 5 and Hailo-8/8L edge deployment.

Based on [blue-smarty/Trainer](https://github.com/blue-smarty/Trainer), adapted specifically for license plate detection.

## Quickstart

### 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note (Raspberry Pi 5):** Install a compatible PyTorch wheel for ARM64 before `pip install -r requirements.txt` if needed. The Ultralytics package depends on PyTorch.

### 2) Create dataset structure

```bash
python scripts/setup_dataset.py --root data/lnpr_dataset --classes license_plate
```

This creates the standard YOLOv8 structure and a `data.yaml` for training.
Populate `data/lnpr_dataset/images/train` and `data/lnpr_dataset/images/val` with your
license plate images, and the corresponding YOLO-format labels in
`data/lnpr_dataset/labels/train` and `data/lnpr_dataset/labels/val`.

### 3) Train (on host or on the Pi)

```bash
python scripts/train.py --data data/lnpr_dataset/data.yaml --model yolov8n.pt --epochs 100 --imgsz 640
```

Optionally pass the LNPR-tuned config:

```bash
python scripts/train.py --data data/lnpr_dataset/data.yaml --model yolov8n.pt \
  --epochs 100 --cfg configs/yolov8-lnpr.yaml
```

### 4) Export to ONNX (for Hailo toolchain)

```bash
python scripts/export_hailo.py --weights runs/detect/lnpr/weights/best.pt --imgsz 640
```

This produces an ONNX file ready for the next step.

### 5) Convert ONNX to HEF (Hailo Dataflow Compiler)

> **Prerequisites:** Install the Hailo SDK (`hailo_sdk_client`) from the
> [Hailo Developer Zone](https://developer.hailo.ai).

```bash
# Hailo-8L (Raspberry Pi 5 AI HAT+)
python scripts/onnx_to_hef.py --onnx runs/detect/lnpr/weights/best.onnx --hw-arch hailo8l

# With calibration images for better INT8 quantization accuracy
python scripts/onnx_to_hef.py --onnx best.onnx --hw-arch hailo8l --calib-path data/calib_images

# If parser errors suggest specific end nodes, pass them explicitly
python scripts/onnx_to_hef.py --onnx best.onnx --hw-arch hailo8l \
  --end-node /model.22/Sigmoid --end-node /model.22/Concat
```

This produces a `.hef` file in the same directory as the ONNX file. Use
`--output-dir` to write it elsewhere.

### 6) Launch the GUI dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard provides a simple interface for:
- creating the LNPR dataset structure (`setup_dataset.py`)
- running LNPR model training (`train.py`)
- exporting to ONNX (`export_hailo.py`)
- converting ONNX to HEF (`onnx_to_hef.py`)

## Dataset Format

License plate images should be annotated in YOLO format: one `.txt` label file per image,
where each line is `<class_id> <cx> <cy> <w> <h>` (all values normalised to `[0, 1]`).

For a single-class LNPR dataset the class id is always `0` (for `license_plate`).

## Hailo-8/8L Notes

- Use the **latest Hailo SDK/Dataflow Compiler** that supports Hailo-8/8L.
- Export to ONNX with `export_hailo.py`, then compile to HEF with `onnx_to_hef.py`.
- Providing representative license plate images with `--calib-path` gives the best INT8 quantization accuracy.
- Follow Hailo's official documentation for runtime deployment.

## Repository Layout

```
configs/
  yolov8-lnpr.yaml
dashboard/
  app.py
  artifacts.py
  validation.py
scripts/
  setup_dataset.py
  train.py
  export_hailo.py
  onnx_to_hef.py
```