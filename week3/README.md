# Week 3 — WaterSeg AI Deployment

### Multispectral Satellite Water Segmentation | Cellula Technologies

## Overview

**WaterSeg AI** is a local web application that detects water regions in multispectral satellite images using the best-performing **ResNet34 U-Net** model selected in Week 2.

The project deploys the trained model through a **Flask REST API** and an interactive web dashboard for image analysis and visualization.

## Model Details

| Property | Details |
|---|---|
| Architecture | ResNet34 U-Net |
| Framework | PyTorch |
| Input | 12 channels × 128 × 128 |
| Output | 128 × 128 binary water mask |
| Prediction Threshold | 0.5 |
| Initialization | `rgb_zero_extra` |
| Selected Seed | 123 |
| Best Validation IoU | ~0.7867 |
| Inference Device | CPU |

The application reuses the original Week 2 preprocessing and normalization statistics.

## Application Features

- **Test Set Gallery:** Browse and search 46 held-out satellite images.
- **External Upload:** Analyze compatible TIFF or NPY images.
- **Original Image:** RGB visualization of the multispectral input.
- **Binary Mask:** White represents predicted water; black represents background.
- **Water Overlay:** Predicted water regions highlighted in cyan.
- **Water Coverage:** Percentage of image pixels predicted as water.
- **Downloads:** Save original previews, masks, and overlays as PNG files.

*Water Coverage is not a measure of model accuracy.*

## Run Locally (Windows)

**Requirements:** Python 3.10+, Windows, and an internet connection for initial setup.

1. Clone or download this repository.
2. Download `best_water_segmentation_model.pth` from the [Week 3 Release](https://github.com/just0neball/Water-Segmentation-using-Multispectral-and-Optical/releases/tag/week3-v1.0.0).
3. Place the checkpoint inside the `week3/` folder, alongside `app.py`.
4. Double-click `run.bat` to install dependencies and start the application.
5. Open **http://127.0.0.1:5000** in your browser.

Keep the terminal window open while using the app.

**Supported inputs:** `.tif`, `.tiff`, `.npy`

Images must contain 12 spectral channels, have a resolution of 128 × 128 pixels, and follow the original dataset's band order and value conventions. Maximum upload size: 20 MB. Standard RGB photographs are not supported.

## REST API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web dashboard |
| GET | `/health` | Model and server status |
| GET | `/gallery` | List 46 test samples |
| POST | `/predict` | Upload an image using multipart field `file` |
| POST | `/predict-sample/<sample_id>` | Predict a gallery sample |

Prediction endpoints return JSON containing water coverage and PNG image data for the original preview, binary mask, and overlay.

## Testing & Results

The deployment was verified through Flask integration tests and local browser testing on Windows.

- All 46 gallery images and previews verified.
- Three gallery predictions matched standalone inference.
- TIFF and NPY uploads tested successfully.
- Original, Mask, and Overlay outputs validated.
- **9/9 API error-handling tests passed.**
- Local interface and download functionality tested successfully.

### Example Test Predictions

| Sample | Water Coverage | IoU | F1-score |
|---|---:|---:|---:|
| 261 | 0.55% | 0.1845 | 0.3116 |
| 161 | 39.70% | 0.7196 | 0.8370 |
| 245 | 55.34% | 0.9317 | 0.9646 |

*These metrics represent individual test samples, not overall test-set performance.*

## Project Files

| Path | Purpose |
|---|---|
| `app.py` | Flask REST API |
| `inference.py` | Model loading and prediction |
| `src/` | Original preprocessing modules |
| `config/` | Training normalization statistics |
| `templates/` & `static/` | Web dashboard |
| `gallery_samples/` | 46 test images |
| `test_results/` | API testing reports |
| `notebooks/` | Week 3 Jupyter notebook |
| `requirements.txt` | Python dependencies |
| `run.bat` | Windows application launcher |

## Resources

- [Week 3 Deployment Notebook](notebooks/)
- [Trained Model — GitHub Release](https://github.com/just0neball/Water-Segmentation-using-Multispectral-and-Optical/releases/tag/week3-v1.0.0)
- [Complete Project — Week 1, 2 & 3](https://github.com/just0neball/Water-Segmentation-using-Multispectral-and-Optical)

---

**WaterSeg AI — Week 3 Deployment**

*Cellula Technologies Internship Project*

*The application is designed for local demonstration using Flask's development server, not public production hosting.*
