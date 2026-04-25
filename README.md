# YOLO-Powered Search Application

A computer vision-based image search engine built with YOLOv11 and Streamlit. It runs object detection inference across an entire image directory, indexes the results as structured metadata, and exposes a query interface that allows users to retrieve images by specific object classes, search logic (OR / AND), and per-class count thresholds.

---

## Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
  - [CPU Setup](#cpu-setup)
  - [GPU Setup](#gpu-setup)
- [Usage](#usage)
  - [Step 1 — Process Images](#step-1--process-images)
  - [Step 2 — Search](#step-2--search)
  - [Step 3 — Export Results](#step-3--export-results)
- [Configuration](#configuration)
- [Screenshots](#screenshots)
- [License](#license)

---

## Overview

Most image datasets are unstructured — querying them by visual content requires custom scripts every time. This application solves that by treating object detection as an indexing step. Once a directory of images is processed, the detection results are stored in a JSON metadata file. From that point forward, searching across thousands of images by class or count becomes instantaneous, with no re-inference needed.

The application supports both pre-trained YOLOv11 weights from Ultralytics and custom-trained model weights, making it adaptable to domain-specific datasets beyond COCO classes.

---

## Demo

> **Add your demo screenshot or screen recording here.**
>
> To add an image: upload your screenshot to the repository under `assets/` (create the folder if it does not exist), then replace the placeholder below with the correct filename.

```
assets/
  demo_search.png
  demo_bbox.png
```

![Application Demo](assets/demo_search.png)

<!-- Replace `assets/demo_search.png` with the actual path to your screenshot. -->
<!-- You can add multiple images by repeating the line below: -->
<!-- ![Bounding Box View](assets/demo_bbox.png) -->

---

## Features

- **Batch inference** — Processes an entire directory of images in a single run using YOLOv11.
- **Metadata persistence** — Detection results are saved to a JSON file, eliminating the need to re-run inference on subsequent searches.
- **Class-based search** — Query images by one or more detected object classes using OR (any match) or AND (all must match) logic.
- **Count threshold filtering** — Optionally restrict results to images containing at most N instances of a given class.
- **Bounding box overlay** — Renders detection boxes directly on result images, with highlighted boxes for matched classes and dimmed boxes for the rest.
- **Configurable grid display** — Adjust the number of result columns (3 to 6) from within the UI.
- **JSON export** — Download the full structured search results as a JSON file for downstream use.
- **Custom model support** — Accepts any `.pt` weight file, including custom-trained YOLOv11 models.

---

## Project Structure

```
Yolo-Powered-Search-Application/
│
├── app.py                        # Main Streamlit application
│
├── src/
│   ├── inference.py              # YOLOv11Inference class — runs detection on image directories
│   └── utils.py                  # Metadata save/load, base64 encoding, class/count utilities
│
├── configs/                      # Configuration files (model paths, inference settings)
│
├── data/
│   └── raw_data/
│       └── coco-val-2017-500/    # Sample dataset (500 images from COCO val 2017)
│
├── requirements.txt              # Python dependencies
├── instruction.txt               # Environment setup reference
└── README.md
```

---

## Requirements

- Python 3.11
- Conda (recommended for environment management)
- CUDA 12.4 compatible GPU (optional, for GPU-accelerated inference)

**Python dependencies:**

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv11 model inference |
| `streamlit` | Web application framework |
| `torch` + `torchvision` | Deep learning backend |
| `Pillow` | Image loading and rendering |
| `opencv-python` | Image processing |
| `numpy` | Numerical operations |
| `pandas` | Data handling |
| `pyyaml` | Configuration parsing |

---

## Installation

### CPU Setup

```bash
conda create -n yolo_image_search python=3.11 -y
conda activate yolo_image_search
pip install -r requirements.txt
```

### GPU Setup

Requires CUDA 12.4. Install PyTorch with CUDA support before the remaining dependencies.

```bash
conda create -n yolo_image_search_gpu python=3.11 -y
conda activate yolo_image_search_gpu
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

**Verify GPU availability after installation:**

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

For CUDA installation guidance:
- Linux: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
- Windows: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/

---

## Usage

Launch the application with:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

### Step 1 — Process Images

Select **"Process new images"** from the top toggle.

- **Image directory path** — Provide the absolute or relative path to a folder containing images (`.jpg`, `.png`, etc.).
- **Model weight path** — Defaults to `yolo11m.pt`. Replace with the path to any custom `.pt` file to use your own trained model.
- Click **"Start inferencing"** to run detection across all images in the directory.

Once complete, the metadata JSON file is saved alongside your image directory. The app displays the save path for reference.

> **Add a screenshot of the processing screen here.**
> Upload the image to `assets/` and update the line below.

![Processing Screen](assets/step1_process.png)

---

### Step 2 — Search

After processing (or loading existing metadata), the **Search Engine** panel becomes available.

**Search mode:**
- **Any of selected (OR)** — Returns images containing at least one of the selected classes.
- **All selected (AND)** — Returns only images where every selected class is present simultaneously.

**Class selection:** Pick one or more classes from the detected classes dropdown.

**Count threshold (optional):** For each selected class, you can optionally set a maximum instance count. For example, setting `person` to `3` will only return images where 3 or fewer persons are detected.

**Display options:**
- Toggle bounding boxes on or off.
- Toggle class highlighting — when enabled, only the matched classes are drawn; all others are hidden.
- Adjust grid column count (3 to 6).

> **Add a screenshot of search results with bounding boxes here.**
> Upload the image to `assets/` and update the line below.

![Search Results](assets/step2_results.png)

---

### Step 3 — Export Results

At the bottom of the results panel, an **Export Options** section allows you to download all matching results as a structured JSON file (`search_results.json`).

The exported file includes image paths, all detections (class, bounding box, confidence), and per-class counts for each matched image.

---

## Configuration

The `configs/` directory is intended for storing inference configuration files. If you are using a custom model trained on non-COCO classes, place your model config or class definition file here and update the model path input in the UI accordingly.

---

## Screenshots

> This section is reserved for a complete visual walkthrough of the application.
> 
> **How to add screenshots:**
> 1. Take screenshots of the running application.
> 2. Create an `assets/` folder in the root of the repository if it does not already exist.
> 3. Upload the images to `assets/`.
> 4. Replace each placeholder below with the correct filename.

| Step | Description | Screenshot |
|---|---|---|
| Processing | Running inference on an image directory | ![](assets/step1_process.png) |
| Search UI | Class selection and threshold controls | ![](assets/step2_search_ui.png) |
| Results grid | Matched images with bounding boxes | ![](assets/step3_results.png) |
| Export | JSON download panel | ![](assets/step4_export.png) |

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.