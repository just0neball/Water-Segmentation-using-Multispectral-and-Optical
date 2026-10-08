"""Week 1 `benchmark12` preprocessing (numpy only, no torch).

Channel layout of the 12-band tensors (0-indexed):
    0 Coastal | 1 Blue | 2 Green | 3 Red | 4 NIR | 5 SWIR1 | 6 SWIR2
    7 QA | 8 MERIT DEM | 9 Copernicus DEM | 10 ESA WorldCover | 11 Water Occurrence

All statistics come from the *training split only* (stats.json, created in Week 1).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6
NUM_CHANNELS = 12
IMAGE_SHAPE = (NUM_CHANNELS, 128, 128)

CHANNEL_NAMES = [
    "Coastal", "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2",
    "QA", "MERIT DEM", "Copernicus DEM", "ESA WorldCover", "Water Occurrence",
]

# Dataset indices of the R, G, B bands (used for visualisation and for the
# `rgb_zero_extra` first-conv initialisation).
RGB_CHANNEL_INDICES = (3, 2, 1)


def preprocess_spectral(image: np.ndarray, stats: dict) -> np.ndarray:
    """Channels 0-6: clip to train P1/P99 then standardise with train mean/std."""
    x = image[:7].astype(np.float32).copy()

    low = np.asarray(stats["spectral_clip_low"], dtype=np.float32)[:, None, None]
    high = np.asarray(stats["spectral_clip_high"], dtype=np.float32)[:, None, None]
    mean = np.asarray(stats["spectral_mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(stats["spectral_std"], dtype=np.float32)[:, None, None]

    x = np.clip(x, low, high)
    x = (x - mean) / (std + EPS)
    return x.astype(np.float32)


def _standardise(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (x - float(mean)) / (float(std) + EPS)


def preprocess_12ch_benchmark(image: np.ndarray, stats: dict) -> np.ndarray:
    """Per-band normalisation of one (12, 128, 128) image -> (12, 128, 128) float32."""
    image = np.asarray(image, dtype=np.float32)
    if image.shape != IMAGE_SHAPE:
        raise ValueError(f"Expected image shape {IMAGE_SHAPE}, got {image.shape}")

    out = np.empty_like(image, dtype=np.float32)

    out[:7] = preprocess_spectral(image, stats)                              # spectral
    out[7] = _standardise(image[7], stats["qa_mean"], stats["qa_std"])       # QA

    merit = image[8].copy()                                                  # MERIT DEM
    copernicus = image[9].copy()
    missing = merit == -9999
    merit[missing] = copernicus[missing]                                     # fill -9999
    out[8] = _standardise(merit, stats["merit_mean"], stats["merit_std"])
    out[9] = _standardise(copernicus, stats["copernicus_mean"], stats["copernicus_std"])

    out[10] = _standardise(image[10], stats["worldcover_mean"], stats["worldcover_std"])

    water_occ = np.clip(image[11], 0, 100)                                   # Water Occurrence
    out[11] = _standardise(water_occ, stats["water_occ_mean"], stats["water_occ_std"])

    if not np.isfinite(out).all():
        raise ValueError("12-channel preprocessing produced NaN or Inf values.")
    return out.astype(np.float32)


def make_rgb_composite(raw_image: np.ndarray) -> np.ndarray:
    """Percentile-stretched RGB composite (visualisation only). Returns (H, W, 3)."""
    r, g, b = RGB_CHANNEL_INDICES
    rgb = np.stack([raw_image[r], raw_image[g], raw_image[b]], axis=-1).astype(np.float32)
    out = np.zeros_like(rgb)
    for c in range(3):
        lo, hi = np.percentile(rgb[..., c], [2, 98])
        out[..., c] = np.clip((rgb[..., c] - lo) / (hi - lo + 1e-6), 0, 1)
    return out
