# Water Segmentation Using Multispectral and Optical Data

Binary water segmentation from multispectral satellite imagery using deep learning in PyTorch.

This project develops and evaluates segmentation models for detecting water pixels from 12-channel satellite image patches. The work includes exploratory data analysis, channel-specific preprocessing, feature engineering, shortcut-risk analysis, U-Net implementation from scratch, model optimization, multi-seed evaluation, and held-out test evaluation.

---

## Project Objective

The main objectives were to:

- Understand the structure and distribution of the multispectral dataset.
- Visualize and analyze all 12 input channels.
- Build a complete preprocessing pipeline using training-derived statistics only.
- Implement U-Net from scratch without pretrained weights.
- Establish a standard 12-channel benchmark.
- Investigate engineered spectral and auxiliary features.
- Improve the benchmark through model architecture and training optimization.
- Evaluate the final model using multiple random seeds and a held-out test set.

---

## Dataset

The dataset contains:

- **306 multispectral images**
- **306 corresponding binary segmentation masks**
- Image size: **128 × 128**
- Number of channels: **12**
- Mask classes:
  - `0` = Background / Non-water
  - `1` = Water

Each image has shape:

```text
(12, 128, 128)
```

### Input Channels

| Channel | Feature |
|---:|---|
| 1 | Coastal Aerosol |
| 2 | Blue |
| 3 | Green |
| 4 | Red |
| 5 | NIR |
| 6 | SWIR1 |
| 7 | SWIR2 |
| 8 | QA Band |
| 9 | MERIT DEM |
| 10 | Copernicus DEM |
| 11 | ESA WorldCover |
| 12 | Water Occurrence Probability |

---

## Exploratory Data Analysis

The notebook includes:

- Image and mask pairing validation
- Shape and dtype inspection
- NaN and infinity checks
- Visualization of all 12 channels
- RGB composites
- Binary mask inspection
- Water-class distribution analysis
- Representative low-, medium-, and high-water samples
- Water versus non-water spectral signatures
- DEM comparison
- Auxiliary-feature shortcut audit
- Spatial leakage investigation

Approximately **26% of all pixels are water pixels**, so pixel accuracy alone is not sufficient for evaluating the segmentation model.

IoU, F1-score, precision, and recall are therefore used as the main segmentation metrics.

---

## Spectral Analysis

The exploratory analysis showed that water is particularly distinguishable in:

- NIR
- SWIR1
- SWIR2

Water generally has much lower reflectance in these bands than many non-water surfaces.

Two explicit water indices were also evaluated:

### NDWI

```text
NDWI = (Green - NIR) / (Green + NIR + ε)
```

### MNDWI

```text
MNDWI = (Green - SWIR1) / (Green + SWIR1 + ε)
```

For the final preprocessing pipeline, reflectance values used for the water indices are constrained to non-negative values before index calculation.

---

## Preprocessing

All normalization statistics are calculated from the **training split only** to avoid validation or test leakage.

The preprocessing pipeline includes:

- Train-derived percentile clipping for spectral bands
- Channel-wise standardization
- Non-negative reflectance handling before NDWI/MNDWI calculation
- MERIT DEM missing-value replacement using Copernicus DEM
- Relative elevation calculation
- WorldCover categorical one-hot encoding
- Water Occurrence preprocessing
- QA bit decomposition for diagnostic experiments

The preprocessing statistics are saved in:

```text
config/stats.json
```

The reproducible data split is stored in:

```text
config/split_v2.csv
```

---

## Shortcut-Risk Analysis

Not all channels are ordinary independent sensor measurements.

The analysis showed that some auxiliary channels contain direct or indirect information related to water presence, particularly:

- QA Band
- Water Occurrence Probability
- ESA WorldCover

Therefore, improvements from these features are interpreted as **auxiliary-prior improvements**, not automatically as improved spectral understanding.

QA showed particularly strong standalone water-related information, so the final selected feature configuration excludes QA bits.

---

## Baseline Model

A U-Net was implemented **from scratch in PyTorch** with no pretrained weights.

The required baseline uses the original normalized **12-channel input**.

The improved U-Net training version includes:

- Batch Normalization
- AdamW optimizer
- BCE + Dice loss
- Learning-rate scheduling
- Longer training
- Early stopping
- Reproducible random seeds

This model is referred to as:

```text
UNetBN + benchmark12
```

---

## Final Model

The selected final architecture is:

```text
ResUNetBN + E4
```

`ResUNetBN` introduces residual convolution blocks while retaining the encoder-decoder segmentation structure.

The E4 input configuration contains:

- Seven spectral bands
- NDWI
- MNDWI
- Imputed elevation
- Relative elevation
- WorldCover one-hot features
- Water Occurrence prior

QA bits are intentionally excluded from the final E4 configuration because of their high shortcut risk.

---

## Improved Training Protocol

The final training pipeline was strengthened after the initial experiments.

Key improvements include:

- Batch Normalization
- Residual convolution blocks
- AdamW optimization
- BCE + Dice segmentation loss
- Learning-rate scheduler
- Extended training duration
- Longer early-stopping patience
- Multiple random seeds
- Held-out test evaluation
- Global and per-image IoU
- Threshold evaluation
- Test-time augmentation using flips

---

## Multi-Seed Evaluation

Core models were evaluated using three random seeds:

```text
42
123
2026
```

This reduces the risk of reporting conclusions based on one favorable random initialization.

The complete multi-seed results are available in:

```text
results/all_multiseed_results.csv
results/all_multiseed_summary.csv
results/core_multiseed_results.csv
results/core_multiseed_summary.csv
```

---

## Final Held-Out Test Results

### Mean Results Across Three Seeds

| Model | Architecture | Test Base IoU | Test TTA IoU | Test TTA F1 | Mean Per-Image IoU |
|---|---|---:|---:|---:|---:|
| Required 12-channel benchmark | UNetBN | 0.6845 ± 0.0055 | **0.6851 ± 0.0064** | 0.8131 | 0.6011 |
| Final engineered model | ResUNetBN + E4 | 0.7405 ± 0.0005 | **0.7413 ± 0.0036** | **0.8514** | **0.6565** |

The selected final model improved mean held-out test TTA IoU by:

```text
+0.0562
```

over the required standard 12-channel benchmark.

---

## Seed-Level Comparison

| Seed | Final ResUNetBN + E4 | 12-Channel Benchmark | Δ IoU |
|---:|---:|---:|---:|
| 42 | 0.7452 | 0.6793 | +0.0659 |
| 123 | 0.7405 | 0.6839 | +0.0566 |
| 2026 | 0.7381 | 0.6920 | +0.0461 |

The final model outperformed the 12-channel benchmark for **all three evaluated seeds**.

---

## Global IoU vs Per-Image IoU

Both global IoU and mean per-image IoU are reported.

Global IoU aggregates all validation or test pixels together and can be influenced more strongly by images containing large water regions.

Mean per-image IoU gives each image equal importance and therefore provides an additional view of model robustness across different scenes.

The final model improved both measures.

---

## Error Analysis

Per-image IoU was calculated to identify difficult samples.

The lowest-performing samples were inspected visually to understand failure modes.

Observed difficult cases include possible:

- Thin water structures
- Turbid or visually ambiguous water
- Water boundaries
- Shadows and dark land surfaces
- Disagreement between the optical image and supplied mask

Some examples suggest possible annotation noise or temporal mismatch between imagery and masks.

This is treated as a **possible limitation**, not as a confirmed labeling error.

---

## Test-Time Augmentation

Flip-based Test-Time Augmentation (TTA) was evaluated consistently for both the benchmark and final model.

TTA produced:

```text
Benchmark:
Base IoU = 0.6845
TTA IoU  = 0.6851

Final:
Base IoU = 0.7405
TTA IoU  = 0.7413
```

The improvement from TTA is small, while the major gain comes from the improved model and feature configuration.

---

## Repository Structure

```text
Water-Segmentation-using-Multispectral-and-Optical/
│
├── water-segmentation-using-multispectral-and-optical.ipynb
│
├── config/
│   ├── stats.json
│   └── split_v2.csv
│
├── results/
│   ├── all_multiseed_results.csv
│   ├── all_multiseed_summary.csv
│   ├── core_multiseed_results.csv
│   ├── core_multiseed_summary.csv
│   ├── final_vs_benchmark_test_results.csv
│   └── history_*.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

Large model checkpoints are intentionally excluded from the repository.

---

## How to Run

The project was developed and trained in Kaggle.

1. Attach the multispectral water-segmentation dataset to the Kaggle notebook.
2. Enable a GPU accelerator.
3. Open:

```text
water-segmentation-using-multispectral-and-optical.ipynb
```

4. Run all cells in order.

The notebook performs:

```text
Dataset Validation
        ↓
EDA
        ↓
Train / Validation / Test Split
        ↓
Preprocessing
        ↓
Feature Engineering
        ↓
Dataset & DataLoader
        ↓
UNetBN Benchmark
        ↓
Multi-Seed Experiments
        ↓
ResUNetBN Final Model
        ↓
Threshold / TTA Evaluation
        ↓
Held-Out Test Evaluation
        ↓
Per-Image Error Analysis
```

---

## Main Libraries

The project uses:

```text
Python
PyTorch
NumPy
Pandas
Matplotlib
Rasterio
Tifffile
Pillow
Scikit-learn
```

---

## Reproducibility

The repository provides:

- Exact data split
- Training-derived preprocessing statistics
- Multi-seed experiment results
- Training histories
- Full executed notebook
- Final held-out benchmark comparison

The reported final results are based on seeds:

```text
42, 123, 2026
```

---

## Limitations

Important limitations include:

- The dataset contains only 306 paired samples.
- Validation and held-out test subsets are therefore relatively small.
- WorldCover and Water Occurrence contain useful auxiliary priors.
- QA contains a particularly strong water-related shortcut signal.
- Source-scene metadata was unavailable, so possible spatial dependence between nearby patches could not be conclusively verified or ruled out.
- Some image-mask pairs may contain temporal mismatch or annotation ambiguity.
- Larger datasets and geographically independent external evaluation would provide stronger evidence of generalization.

---

## Conclusion

A standard normalized 12-channel U-Net benchmark achieved:

```text
Test TTA IoU = 0.6851 ± 0.0064
```

The final **ResUNetBN + E4** configuration achieved:

```text
Test TTA IoU = 0.7413 ± 0.0036
Test TTA F1  = 0.8514
```

This corresponds to an average held-out Test IoU improvement of:

```text
+0.0562
```

The improvement was positive for all three evaluated seeds.

The results suggest that combining stronger model optimization, residual segmentation blocks, engineered spectral information, and carefully selected auxiliary priors provides a meaningful improvement over the standard 12-channel benchmark.
