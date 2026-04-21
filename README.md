# ULite - Lightweight Surface Defect Segmentation on NEU-seg

PyTorch implementation of the XULite model for steel surface defect segmentation (binary segmentation) on the NEU-seg dataset.

Main notebook: `XUlite.ipynb`

## Table of Contents

1. Overview
2. Model Architecture
3. Experimental Results
4. Folder Structure
5. Environment Requirements
6. Data Preparation
7. Training and Evaluation
8. Quick Start
9. Future Improvements
10. Citation

## 1. Overview

- Task: binary segmentation (defect vs. background)
- Input: `200x200` grayscale images
- Output: `1`-channel binary masks
- Framework: PyTorch

## 2. Model Architecture

The model uses a U-shaped encoder-decoder architecture with skip connections, combined with the following building blocks:

- `XConv`: depthwise convolution with main-diagonal and anti-diagonal masks
- `AxialDW`: depthwise convolution along horizontal and vertical axes
- `3-branch bottleneck`: `AxialDW + XConv + DWConv 3x3`

## 3. Experimental Results

### Training Configuration

| Component | Value |
| --- | --- |
| Device | CUDA |
| Epochs | 50 |
| Batch size | 32 |
| Learning rate | `1e-3` |
| Optimizer | AdamW |
| Loss | BCEWithLogitsLoss |
| LR Scheduler | ReduceLROnPlateau |
| Checkpoint | `best_xulite_model.pth` |

Training time: about `36 minutes 31 seconds` (`10:07:53 -> 10:44:24`).

### Training Results

| Metric | Value |
| --- | --- |
| Train Loss | `0.3258 -> 0.0463` |
| Train Accuracy | `83.36% -> 98.03%` |
| Validation Loss | `0.1659 -> 0.0559` |
| Best Validation Loss | `0.0551` at epoch `47` |
| Final Validation Accuracy | `97.74%` |

### Segmentation Metrics on Validation Set

| Metric | Score |
| --- | --- |
| Precision | `0.9285` |
| Recall | `0.8863` |
| F1-score | `0.9069` |
| mIoU | `0.8297` |

### Inference Performance

| Metric | Value |
| --- | --- |
| Time for 1000 images | `7.082` seconds |
| FPS | `141.20` img/s |

### Model Complexity

| Metric | Value |
| --- | --- |
| FLOPs | `229.327M` |
| Parameters (THOP) | `864.545K` |
| Parameters (counted from model) | `979,009` (`0.979M`) |

Note: small differences in parameter counts may appear across profiling tools due to different counting methods.

## 4. Folder Structure

```text
XUlite/
|-- XUlite.ipynb
|-- README.md
|-- train.npz
|-- val.npz
|-- datasets/
`-- NEU-seg/
    |-- TrainingData/
    |-- ValData/
    `-- TestData/
```

## 5. Environment Requirements

- Python `3.10+`
- CUDA if an NVIDIA GPU is available

Install the required libraries with:

```bash
pip install torch torchvision opencv-python numpy pandas matplotlib tqdm thop
```

## 6. Data Preparation

The notebook already provides:

- `rle_decode(...)`: decode masks from RLE format
- `build_npz(...)`: convert images and RLE CSV labels into `.npz` files

Example for building the train and validation sets:

```python
build_npz(train_dr, train_csv_path, "train.npz", "train")
build_npz(val_dr, val_csv_path, "val.npz", "val")
```

The `SurfaceDefectDatasetV2` loader includes:

- CLAHE for grayscale contrast enhancement
- Augmentation: horizontal flip, vertical flip, rotation
- Normalization with `mean=[0.5]` and `std=[0.5]`

## 7. Training and Evaluation

The notebook already includes the full pipeline:

- training and validation by epoch
- saving the best checkpoint based on validation loss
- computing Precision, Recall, F1-score, and mIoU
- measuring FPS, FLOPs, and parameter count

If you run locally instead of Colab, update the `/content/drive/...` paths to match your local machine.

## 8. Quick Start

1. Open `XUlite.ipynb`.
2. Update the dataset paths.
3. Run the cells in order: model -> dataloader -> train -> evaluate -> visualize.
4. Check the checkpoint file `best_xulite_model.pth`.

## 9. Future Improvements

- Split the notebook into modules such as `model.py`, `dataset.py`, `train.py`, and `eval.py`
- Add `requirements.txt` and CLI scripts for easier experiment reproduction
- Add Dice score and benchmark on an independent test set

## 10. Citation

Reference used:

- ScienceDirect article (PII: `S1051200425006578`): https://www.sciencedirect.com/science/article/pii/S1051200425006578
- Access date: `2026-04-08`
