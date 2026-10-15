"""Dataset, augmentation and reproducible DataLoaders.

The Week 1 train / validation / test split (`split_v2.csv`) and the train-derived
preprocessing statistics (`stats.json`) are reused unchanged.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .preprocessing import IMAGE_SHAPE, preprocess_12ch_benchmark
from .utils import seed_everything

# --------------------------------------------------------------------------
# File access
# --------------------------------------------------------------------------


def load_raw_image(path) -> np.ndarray:
    """Load one multispectral TIFF -> float32 array of shape (12, 128, 128)."""
    image = tifffile.imread(path)
    if image.shape == (128, 128, 12):          # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))
    if image.shape != IMAGE_SHAPE:
        raise ValueError(f"Unexpected image shape {image.shape} for {Path(path).name}")
    return image.astype(np.float32)


def load_binary_mask(path) -> np.ndarray:
    """Load one water mask -> float32 array (128, 128) with values {0, 1}."""
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape != IMAGE_SHAPE[1:]:
        raise ValueError(f"Unexpected mask shape {mask.shape} for {Path(path).name}")
    return (mask > 0).astype(np.float32)


def build_file_maps(images_dir, masks_dir):
    """Map sample id (file stem) -> path for images and masks; ids must match."""
    image_map = {p.stem: p for p in sorted(Path(images_dir).glob("*.tif"))}
    mask_map = {p.stem: p for p in sorted(Path(masks_dir).glob("*.png"))}
    if set(image_map) != set(mask_map):
        raise AssertionError("Image and mask IDs do not match.")
    return image_map, mask_map


def load_stats(stats_path) -> dict:
    with open(stats_path, "r") as f:
        return json.load(f)


def load_split(split_path, expected_sizes=(214, 46, 46)):
    """Return (full_df, train_df, val_df, test_df) from the Week 1 split file."""
    split_df = pd.read_csv(split_path)
    frames = [split_df[split_df["Split"] == s].reset_index(drop=True)
              for s in ("Train", "Validation", "Test")]
    if expected_sizes is not None:
        sizes = tuple(len(f) for f in frames)
        if sizes != tuple(expected_sizes):
            raise AssertionError(f"Unexpected split sizes {sizes}, expected {tuple(expected_sizes)}")
    return (split_df, *frames)


# --------------------------------------------------------------------------
# Augmentation + Dataset
# --------------------------------------------------------------------------


def apply_spatial_augmentation(image: np.ndarray, mask: np.ndarray):
    """Identical random flips / 90-degree rotations for image (C,H,W) and mask (H,W).

    Uses Python's global `random` module, so DataLoader `num_workers` must stay 0
    for runs to be reproducible.
    """
    if random.random() < 0.5:
        image, mask = np.flip(image, axis=2), np.flip(mask, axis=1)
    if random.random() < 0.5:
        image, mask = np.flip(image, axis=1), np.flip(mask, axis=0)
    k = random.randint(0, 3)
    if k:
        image = np.rot90(image, k=k, axes=(1, 2))
        mask = np.rot90(mask, k=k, axes=(0, 1))
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class WaterSegmentationDataset(Dataset):
    """Yields (image[12,128,128], mask[1,128,128], sample_id).

    Preprocessing is deterministic, so the normalised arrays are cached after the
    first read; augmentation (training only) is applied on a copy every epoch.
    """

    def __init__(self, dataframe, image_map, mask_map, stats, augment=False, cache=True):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.image_map, self.mask_map = image_map, mask_map
        self.stats, self.augment, self.cache = stats, augment, cache
        self._cache: dict = {}

    def __len__(self):
        return len(self.dataframe)

    def _load(self, sample_id: str):
        if self.cache and sample_id in self._cache:
            return self._cache[sample_id]
        image = preprocess_12ch_benchmark(load_raw_image(self.image_map[sample_id]), self.stats)
        mask = load_binary_mask(self.mask_map[sample_id])
        if self.cache:
            self._cache[sample_id] = (image, mask)
        return image, mask

    def get_processed(self, sample_id):
        """Normalised (image[12,H,W], mask[H,W]) for one sample id (no augmentation)."""
        image, mask = self._load(str(int(sample_id)))
        return image.copy(), mask.copy()

    def __getitem__(self, index):
        sample_id = str(int(self.dataframe.loc[index, "Sample_ID"]))
        image, mask = self._load(sample_id)
        image, mask = image.copy(), mask.copy()
        if self.augment:
            image, mask = apply_spatial_augmentation(image, mask)
        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).float()
        return image, mask, sample_id


def build_datasets(train_df, val_df, test_df, image_map, mask_map, stats):
    """Create the three datasets once; reuse them for every seed (shared cache)."""
    return {
        "train": WaterSegmentationDataset(train_df, image_map, mask_map, stats, augment=True),
        "val": WaterSegmentationDataset(val_df, image_map, mask_map, stats, augment=False),
        "test": WaterSegmentationDataset(test_df, image_map, mask_map, stats, augment=False),
    }


def make_dataloaders(datasets, batch_size: int, seed: int, num_workers: int = 0):
    """Deterministic loaders: the training shuffle order depends only on `seed`."""
    seed_everything(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    pin = torch.cuda.is_available()

    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, generator=generator)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader
