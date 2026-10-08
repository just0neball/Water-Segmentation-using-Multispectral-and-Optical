"""Experiment configuration (one place for every hyper-parameter and path)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

WEEK1_RESULTS_URL = (
    "https://raw.githubusercontent.com/just0neball/"
    "Water-Segmentation-using-Multispectral-and-Optical/main/results/core_multiseed_results.csv"
)


@dataclass
class PipelineConfig:
    # ---- paths -----------------------------------------------------------
    data_root: Path                       # folder containing images/ and labels/
    config_dir: Path                      # folder containing split_v2.csv and stats.json
    out_dir: Path                         # where checkpoints / results / figures are written
    readme_path: Optional[Path] = None    # results table is injected here if given

    # ---- experiment grid -------------------------------------------------
    seeds: Tuple[int, ...] = (42, 123, 2026)
    strategies: Tuple[str, ...] = ("mean", "rgb_zero_extra")
    ablation_random_init: bool = True     # same architecture, no ImageNet weights
    encoder: str = "resnet34"
    decoder: str = "unet"                 # "unet" or "deeplabv3plus"
    in_channels: int = 12

    # ---- training --------------------------------------------------------
    batch_size: int = 16
    max_epochs: int = 100
    patience: int = 15
    encoder_lr: float = 1e-4
    decoder_lr: float = 1e-3
    weight_decay: float = 1e-4
    threshold: float = 0.5
    num_workers: int = 0                  # keep 0: augmentation uses Python's global RNG
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    dice_mode: str = "batch"              # "batch" or "per_image" (must match Week 1)
    log_every: int = 5
    resume: bool = True                   # skip runs whose checkpoint already exists

    # ---- Week 1 baseline (for the comparison) ----------------------------
    week1_results: Optional[str] = None   # path / URL of core_multiseed_results.csv
    week1_test_csv: Optional[str] = None  # optional: Week 1 test metrics per seed
    week1_params_m: Optional[float] = None  # optional: Week 1 parameter count (millions)

    # ---- derived paths ---------------------------------------------------
    @property
    def images_dir(self) -> Path:
        return Path(self.data_root) / "images"

    @property
    def masks_dir(self) -> Path:
        return Path(self.data_root) / "labels"

    @property
    def split_path(self) -> Path:
        return Path(self.config_dir) / "split_v2.csv"

    @property
    def stats_path(self) -> Path:
        return Path(self.config_dir) / "stats.json"

    def to_dict(self) -> dict:
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}
