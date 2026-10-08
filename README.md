# Water Segmentation Using Multispectral and Optical Data

This repository contains a two-week water segmentation project using multispectral satellite imagery.

## Week 1 — Water Segmentation from Scratch

Week 1 focuses on building and evaluating a water-segmentation pipeline from scratch.

Main work:
- Dataset exploration and preprocessing
- 12-channel multispectral input
- U-Net trained from scratch
- Multi-seed evaluation
- Held-out test evaluation

➡️ [Open Week 1](./week1/)

---

## Week 2 — Pretrained Multispectral Water Segmentation

Week 2 extends the project using transfer learning.

Main work:
- ImageNet-pretrained ResNet34 encoder
- U-Net decoder
- Adaptation from 3 RGB channels to 12 channels
- First-layer initialization comparison
- Random-init architecture control
- Multi-seed fine-tuning
- Comparison with the Week 1 scratch model

➡️ [Open Week 2](./week2/)

---

## Final Week 2 Result

The selected pretrained ResNet34-U-Net achieved:

- **Test IoU:** 0.7440 ± 0.0031
- **Week 1 scratch Test IoU:** 0.6845 ± 0.0055
- **Selected initialization:** `rgb_zero_extra`

## Repository Structure

```text
├── week1/
├── week2/
├── .gitignore
└── README.md