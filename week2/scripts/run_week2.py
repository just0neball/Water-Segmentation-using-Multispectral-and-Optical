#!/usr/bin/env python
"""Run the whole Week 2 experiment from the command line.

Examples
--------
Full run (3 seeds, both init strategies + random-init ablation), on Kaggle:
    python scripts/run_week2.py --data-root /kaggle/input/datasets/nebalelshobary/satellite-multispectral-water-segmentation

Quick smoke test (1 seed, 2 epochs):
    python scripts/run_week2.py --data-root <path> --quick

Optional extras:
    --decoder deeplabv3plus     # DeepLabV3+ decoder instead of U-Net
    --week1-test-csv week1_test_metrics.csv --week1-params-m 7.8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import PipelineConfig  # noqa: E402
from src.pipeline import run_pipeline   # noqa: E402

KAGGLE_DATA = "/kaggle/input/datasets/nebalelshobary/satellite-multispectral-water-segmentation"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=KAGGLE_DATA if Path(KAGGLE_DATA).exists() else None,
                   help="folder containing images/ and labels/")
    p.add_argument("--config-dir", default=str(REPO_ROOT / "config"), help="folder with split_v2.csv and stats.json")
    p.add_argument("--out-dir", default=str(REPO_ROOT), help="output root (checkpoints/, results/, figures/)")
    p.add_argument("--readme", default=str(REPO_ROOT / "README.md"), help="README whose RESULTS block is refreshed ('' to skip)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    p.add_argument("--strategies", nargs="+", default=["mean", "rgb_zero_extra"],
                   choices=["mean", "rgb_zero_extra"])
    p.add_argument("--no-ablation", action="store_true", help="skip the random-init (no ImageNet) control runs")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--decoder", default="unet", choices=["unet", "deeplabv3plus"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--encoder-lr", type=float, default=1e-4)
    p.add_argument("--decoder-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dice-mode", default="batch", choices=["batch", "per_image"],
                   help="must match the Week 1 loss for a like-for-like comparison")
    p.add_argument("--week1-results", default=None, help="Week 1 core_multiseed_results.csv (path or URL)")
    p.add_argument("--week1-test-csv", default=None, help="optional CSV with Week 1 test metrics per seed")
    p.add_argument("--week1-params-m", type=float, default=None, help="Week 1 model size in millions of parameters")
    p.add_argument("--no-resume", action="store_true", help="retrain even if checkpoints exist")
    p.add_argument("--quick", action="store_true", help="smoke test: 1 seed, 2 epochs, no ablation")
    return p.parse_args()


def main():
    a = parse_args()
    if not a.data_root:
        sys.exit("error: --data-root is required (folder containing images/ and labels/)")

    cfg = PipelineConfig(
        data_root=Path(a.data_root), config_dir=Path(a.config_dir), out_dir=Path(a.out_dir),
        readme_path=Path(a.readme) if a.readme else None,
        seeds=tuple(a.seeds), strategies=tuple(a.strategies), ablation_random_init=not a.no_ablation,
        encoder=a.encoder, decoder=a.decoder, batch_size=a.batch_size, max_epochs=a.max_epochs,
        patience=a.patience, encoder_lr=a.encoder_lr, decoder_lr=a.decoder_lr, weight_decay=a.weight_decay,
        dice_mode=a.dice_mode, week1_results=a.week1_results, week1_test_csv=a.week1_test_csv,
        week1_params_m=a.week1_params_m, resume=not a.no_resume,
    )
    if a.quick:
        cfg.seeds, cfg.max_epochs, cfg.patience, cfg.ablation_random_init = (cfg.seeds[0],), 2, 2, False
        cfg.out_dir = Path(a.out_dir) / "_quick_run"
        cfg.readme_path = None            # never overwrite the real README with smoke-test numbers
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
