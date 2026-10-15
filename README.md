# Water Segmentation Using Multispectral and Optical Data

### Deep Learning • Satellite Imagery • Semantic Segmentation • Model Deployment

An end-to-end deep learning project for **water segmentation from 12-channel multispectral satellite imagery**, developed across three stages: training a segmentation model from scratch, improving performance through transfer learning, and deploying the selected model as an interactive web application.

The project covers the complete machine learning workflow, from data exploration and model experimentation to evaluation, inference, REST API development, and local deployment.

---

## Project Overview

**Objective:** Identify and segment water regions in multispectral satellite images.

**Dataset:** Multispectral satellite imagery with 12 spectral channels and corresponding binary water masks.

**Input:** 12 × 128 × 128 multispectral image

**Output:** 128 × 128 binary segmentation mask

- White pixels represent predicted water.
- Black pixels represent background.

**Frameworks & Technologies:** Python, PyTorch, segmentation-models-pytorch, NumPy, Flask, HTML, CSS, and JavaScript.

---

## Project Development Roadmap

| Stage | Focus | Key Outcome |
|---|---|---|
| Week 1 | Model Development from Scratch | Custom U-Net segmentation pipeline |
| Week 2 | Transfer Learning & Optimization | ResNet34 U-Net with improved test performance |
| Week 3 | Deployment & Application Development | Interactive WaterSeg AI web application |

---

## Week 1 — Water Segmentation from Scratch

The first stage focuses on developing a complete water segmentation pipeline without relying on pretrained model weights.

### Main Work

- Dataset exploration and preprocessing
- Handling 12-channel multispectral satellite images
- U-Net architecture implemented from scratch
- Training and validation pipeline using PyTorch
- Multi-seed experimentation
- Evaluation using IoU, F1-score, precision, and recall
- Held-out test evaluation

### Outcome

A fully trained scratch-based segmentation model, establishing the baseline performance for subsequent experiments.

**Reported Test IoU:** 0.6845 ± 0.0055

➡️ [Explore Week 1 — Implementation & Results](week1/)

---

## Week 2 — Pretrained Multispectral Water Segmentation

The second stage improves the segmentation pipeline using transfer learning and systematic experimentation.

### Main Work

- ImageNet-pretrained ResNet34 encoder
- U-Net segmentation decoder
- Adaptation from 3 RGB channels to 12 multispectral channels
- Comparison of first-layer initialization strategies
- Random-initialization architecture control
- Multi-seed fine-tuning
- Training and validation performance analysis
- Comparison against the Week 1 scratch baseline

### Selected Configuration

| Component | Selection |
|---|---|
| Architecture | ResNet34 U-Net |
| Input Channels | 12 |
| Initialization Strategy | `rgb_zero_extra` |
| Selected Checkpoint Seed | 123 |
| Selected Checkpoint Best Epoch | 61 |
| Best Validation IoU of Selected Checkpoint | Approximately 0.7867 |

### Outcome

Transfer learning improved segmentation performance over the original scratch-trained model.

➡️ [Explore Week 2 — Transfer Learning Experiments](week2/)

---

## Model Performance Comparison

The reported Week 2 test results demonstrate an improvement over the Week 1 baseline.

| Model | Reported Test IoU |
|---|---:|
| Week 1 — U-Net from Scratch | 0.6845 ± 0.0055 |
| Week 2 — Pretrained ResNet34 U-Net | **0.7440 ± 0.0031** |

**Best-performing initialization strategy:** `rgb_zero_extra`

The Week 2 model was selected for deployment in Week 3.

---

## Week 3 — WaterSeg AI: Model Deployment

The final stage transforms the selected trained segmentation model into a functional application that users can interact with locally.

### WaterSeg AI — Interactive Web Application

WaterSeg AI is a Flask-based web application that performs inference on multispectral satellite imagery and presents segmentation results through a responsive dashboard.

### Key Features

- **46-Image Test Gallery:** Browse and select images from the held-out test dataset.
- **Image Search:** Find gallery samples by sample number.
- **External Image Upload:** Upload compatible TIFF, TIFF, or NPY files.
- **Original Satellite Image:** Display an RGB visualization of the multispectral input.
- **Predicted Water Mask:** Display the binary water segmentation output.
- **Water Detection Overlay:** Highlight predicted water regions in cyan over the original RGB preview.
- **Water Coverage:** Calculate the percentage of pixels classified as water.
- **Download Results:** Export original previews, binary masks, and overlays as PNG files.
- **REST API:** Perform segmentation through Flask endpoints.
- **Local Windows Deployment:** Run the application locally using the provided launcher.

### Application Workflow

**Select or Upload Image → Multispectral Preprocessing → ResNet34 U-Net Inference → Binary Water Mask → Visualization & Download**

The original 12-channel preprocessing and training normalization statistics are reused during inference.

### Week 3 Integration Testing

The deployment pipeline was validated through:

- Verification of all 46 test gallery images and RGB previews
- Three real gallery prediction tests
- Consistency checks against standalone model inference
- Successful external TIFF and NPY uploads
- Verification of original, mask, and overlay outputs
- Nine successful API error-handling tests
- Local browser testing on Windows

### Example Inference Results

| Test Sample | Predicted Water Coverage | IoU | F1-score |
|---|---:|---:|---:|
| Sample 261 | 0.55% | 0.1845 | 0.3116 |
| Sample 161 | 39.70% | 0.7196 | 0.8370 |
| Sample 245 | 55.34% | 0.9317 | 0.9646 |

These represent individual sample results, not the overall test-set performance.

**Important:** Predicted Water Coverage represents the proportion of image pixels classified as water. It is not the model's accuracy or IoU score.

➡️ [Explore Week 3 — WaterSeg AI Deployment](week3/)

➡️ [View Week 3 Notebook](week3/notebooks/)

➡️ [Download Trained Model Checkpoint](https://github.com/just0neball/Water-Segmentation-using-Multispectral-and-Optical/releases/tag/week3-v1.0.0)

---

## Run WaterSeg AI Locally

The Week 3 application can be executed on a Windows computer.

### Setup

1. Clone or download this repository.
2. Open the `week3/` directory.
3. Download `best_water_segmentation_model.pth` from the [Week 3 GitHub Release](https://github.com/just0neball/Water-Segmentation-using-Multispectral-and-Optical/releases/tag/week3-v1.0.0).
4. Place the downloaded checkpoint directly inside `week3/`, alongside `app.py`.
5. Double-click `run.bat`.
6. Wait for the initial dependency installation and model loading.
7. Open `http://127.0.0.1:5000` in your browser.

The first launch requires an internet connection to install dependencies.

### Supported Inputs

- `.tif`
- `.tiff`
- `.npy`

Images must contain 12 spectral channels, have a spatial resolution of 128 × 128 pixels, and follow the training dataset's band order and preprocessing assumptions.

Ordinary RGB photographs are not supported.

The application runs locally using a Flask development server and is not configured for public production hosting.

For full setup instructions, visit [Week 3 Documentation](week3/README.md).

---

## Repository Structure

| Directory / File | Description |
|---|---|
| `week1/` | Scratch U-Net implementation, notebooks, experiments, and evaluation results |
| `week2/` | Transfer learning pipeline, pretrained model experiments, comparisons, and evaluation |
| `week3/` | Flask API, inference pipeline, gallery, frontend, test reports, and deployment utilities |
| `week3/notebooks/` | Week 3 deployment notebook |
| `week3/gallery_samples/` | 46 held-out test satellite images |
| `week3/static/` | CSS, JavaScript, and gallery preview images |
| `week3/templates/` | Web application HTML template |
| `week3/test_results/` | API and integration test results |
| `README.md` | General project overview |

---

## Key Takeaways

This three-week project demonstrates a complete deep learning development workflow:

**Week 1:** Establish a segmentation baseline through training from scratch.

**Week 2:** Improve segmentation performance using pretrained features and systematic experimentation.

**Week 3:** Transform the selected model into a working, locally deployable application with a REST API and interactive visualizations.

The final result connects model research, quantitative evaluation, and practical software deployment within a single reproducible project.

---

## Project Release

**WaterSeg AI — Week 3 Deployment v1.0.0**

[View Release and Download Model Weights](https://github.com/just0neball/Water-Segmentation-using-Multispectral-and-Optical/releases/tag/week3-v1.0.0)

Developed as part of the Cellula Technologies internship project.
