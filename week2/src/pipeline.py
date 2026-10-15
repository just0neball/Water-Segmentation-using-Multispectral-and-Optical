"""End-to-end Week 2 experiment pipeline.

1. Train every (init strategy x seed) run (+ optional random-init ablation), early stopping on val IoU.
2. Pick the best pretrained first-conv strategy by *mean validation IoU over all seeds*.
3. Evaluate every checkpoint once on the held-out test set (after the selection).
4. Compare with the Week 1 scratch U-Net, write CSVs / figures / RESULTS.md (and the README block).
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import pandas as pd
import torch

from . import analysis as an
from . import reporting as rp
from .config import WEEK1_RESULTS_URL, PipelineConfig
from .data import build_datasets, build_file_maps, load_split, load_stats, make_dataloaders
from .engine import (evaluate_model, fit_model, load_checkpoint, predict_sample_counts,
                     read_checkpoint_meta, save_checkpoint)
from .losses import BCEDiceLoss
from .model import INIT_RANDOM, build_model
from .utils import count_parameters, get_device, seed_everything

RANDOM_LABEL = "Random init"
WEEK1_LABEL = "Week 1 scratch"
PRETRAINED_LABEL = "Pretrained"


def run_name(cfg: PipelineConfig, strategy: str, seed: int) -> str:
    return f"{cfg.encoder}_{cfg.decoder}_{strategy}_seed{seed}"


def _free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _make_dirs(cfg: PipelineConfig) -> dict:
    out = Path(cfg.out_dir)
    dirs = {"root": out, "checkpoints": out / "checkpoints", "results": out / "results", "figures": out / "figures"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def train_or_load_run(cfg, strategy, seed, datasets, criterion, device, dirs) -> tuple:
    """Train one run (or reuse an existing checkpoint). Returns (summary_row, history_df)."""
    name = run_name(cfg, strategy, seed)
    ckpt_path = dirs["checkpoints"] / f"{name}.pth"
    hist_path = dirs["results"] / f"history_{name}.csv"

    if cfg.resume and ckpt_path.exists() and hist_path.exists():
        meta = read_checkpoint_meta(ckpt_path)
        print(f"[resume] {name}: using existing checkpoint (best epoch {meta['best_epoch']})")
        return meta["summary_row"], pd.read_csv(hist_path)

    print(f"\n=== {name} ===")
    seed_everything(seed)
    train_loader, val_loader, _ = make_dataloaders(datasets, cfg.batch_size, seed, cfg.num_workers)
    seed_everything(seed)                       # same RNG state right before building the model
    model = build_model(cfg.encoder, cfg.decoder, strategy, cfg.in_channels)

    model, history, best_epoch, _ = fit_model(
        model, train_loader, val_loader, criterion, device, seed=seed,
        max_epochs=cfg.max_epochs, patience=cfg.patience, encoder_lr=cfg.encoder_lr,
        decoder_lr=cfg.decoder_lr, weight_decay=cfg.weight_decay, threshold=cfg.threshold,
        log_every=cfg.log_every)

    val = evaluate_model(model, val_loader, criterion, device, cfg.threshold)
    row = {"Run": name, "Strategy": strategy, "Seed": int(seed), "Best_Epoch": int(best_epoch),
           "Val_IoU": float(val["IoU"]), "Val_F1": float(val["F1"]),
           "Val_Precision": float(val["Precision"]), "Val_Recall": float(val["Recall"]),
           "Val_Loss": float(val["Loss"]), "Params_M": count_parameters(model) / 1e6}

    history.to_csv(hist_path, index=False)
    save_checkpoint(ckpt_path, model, {"seed": int(seed), "strategy": strategy, "encoder": cfg.encoder,
                                       "decoder": cfg.decoder, "in_channels": cfg.in_channels,
                                       "best_epoch": int(best_epoch), "summary_row": row})
    del model, train_loader, val_loader
    _free_memory()
    return row, history


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------


def _load_week1(cfg: PipelineConfig, dirs: dict):
    local_copy = dirs["results"] / "week1_core_multiseed_results.csv"
    source = cfg.week1_results or (str(local_copy) if local_copy.exists() else WEEK1_RESULTS_URL)
    raw = pd.read_csv(source)
    raw.to_csv(local_copy, index=False)             # keep a local copy -> reproducible offline
    week1_val = an.load_week1_baseline(local_copy)
    week1_test = an.load_week1_test_baseline(cfg.week1_test_csv) if cfg.week1_test_csv else None
    return week1_val, week1_test


def run_pipeline(cfg: PipelineConfig, device=None, show_figures: bool = False) -> dict:
    device = device or get_device()
    dirs = _make_dirs(cfg)
    (dirs["root"] / "config_used.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    # ---- data ------------------------------------------------------------
    image_map, mask_map = build_file_maps(cfg.images_dir, cfg.masks_dir)
    split_df, train_df, val_df, test_df = load_split(cfg.split_path)
    stats = load_stats(cfg.stats_path)
    datasets = build_datasets(train_df, val_df, test_df, image_map, mask_map, stats)
    criterion = BCEDiceLoss(cfg.bce_weight, cfg.dice_weight, dice_mode=cfg.dice_mode)
    print(f"Device: {device} | pairs: {len(image_map)} | train/val/test: "
          f"{len(train_df)}/{len(val_df)}/{len(test_df)}")

    # ---- 1. train every run ---------------------------------------------
    strategies = list(cfg.strategies) + ([INIT_RANDOM] if cfg.ablation_random_init else [])
    rows, histories = [], {}
    for strategy in strategies:
        for seed in cfg.seeds:
            row, hist = train_or_load_run(cfg, strategy, seed, datasets, criterion, device, dirs)
            rows.append(row)
            histories[(strategy, seed)] = hist
    runs = pd.DataFrame(rows)
    runs.to_csv(dirs["results"] / "all_runs_validation.csv", index=False)

    # ---- 2. select the first-conv strategy on validation only ------------
    pre = runs[runs["Strategy"].isin(cfg.strategies)]
    strategy_summary = (pre.groupby("Strategy")["Val_IoU"].agg(["mean", "std"])
                        .rename(columns={"mean": "Val_IoU_mean", "std": "Val_IoU_std"})
                        .sort_values("Val_IoU_mean", ascending=False).reset_index())
    selected = str(strategy_summary.loc[0, "Strategy"])
    order = list(strategy_summary["Strategy"])
    pivot = pre.pivot(index="Strategy", columns="Seed", values="Val_IoU").reindex(order)
    pivot.columns = [f"Seed {c}" for c in pivot.columns]
    pivot["Mean ± Std"] = [an.fmt_mean_std(pre[pre["Strategy"] == s]["Val_IoU"]) for s in order]
    strategy_table = pivot.reset_index()
    strategy_table.to_csv(dirs["results"] / "init_strategy_comparison.csv", index=False)
    print(f"\nSelected first-conv initialisation (mean val IoU over seeds): {selected}")

    # ---- 3. held-out test evaluation (after selection) -------------------
    # Pretrained and random-init models share one architecture, so inference models are built
    # without downloading ImageNet weights; the checkpoint overwrites all parameters anyway.
    test_rows, counts = [], []
    for _, r in runs.iterrows():
        model = build_model(cfg.encoder, cfg.decoder, INIT_RANDOM, cfg.in_channels).to(device)
        load_checkpoint(dirs["checkpoints"] / f"{r['Run']}.pth", model, device)
        _, val_loader, test_loader = make_dataloaders(datasets, cfg.batch_size, int(r["Seed"]), cfg.num_workers)
        t = evaluate_model(model, test_loader, criterion, device, cfg.threshold)
        tc = predict_sample_counts(model, test_loader, device, cfg.threshold)
        vc = predict_sample_counts(model, val_loader, device, cfg.threshold)
        test_rows.append({"Run": r["Run"], "Strategy": r["Strategy"], "Seed": int(r["Seed"]),
                          **{f"Test_{m}": float(t[m]) for m in an.METRICS}, "Test_Loss": float(t["Loss"]),
                          "Test_MeanImageIoU_waterOnly": an.mean_image_iou_water_only(tc),
                          "Val_MeanImageIoU_waterOnly": an.mean_image_iou_water_only(vc)})
        for split, c in (("val", vc), ("test", tc)):
            counts.append(c.assign(Run=r["Run"], Strategy=r["Strategy"], Seed=int(r["Seed"]), Split=split))
        del model
        _free_memory()
    test_runs = pd.DataFrame(test_rows)
    counts_df = pd.concat(counts, ignore_index=True)
    test_runs.to_csv(dirs["results"] / "all_runs_test.csv", index=False)
    counts_df.to_csv(dirs["results"] / "per_sample_counts.csv", index=False)

    # ---- 4. comparison with Week 1 ---------------------------------------
    week1_val, week1_test = _load_week1(cfg, dirs)
    def val_plain(strategy):
        return an.wide_to_plain(runs[runs["Strategy"] == strategy], "Val")

    def test_plain(strategy):
        return an.wide_to_plain(test_runs[test_runs["Strategy"] == strategy], "Test")

    has_rand = cfg.ablation_random_init

    val_by_model = {"week1": week1_val, "pretrained": val_plain(selected),
                    "random": val_plain(INIT_RANDOM) if has_rand else None}
    test_by_model = {"week1": week1_test, "pretrained": test_plain(selected),
                     "random": test_plain(INIT_RANDOM) if has_rand else None}
    params = {"week1": cfg.week1_params_m,
              "pretrained": float(runs[runs["Strategy"] == selected]["Params_M"].iloc[0]),
              "random": float(runs[runs["Strategy"] == INIT_RANDOM]["Params_M"].iloc[0]) if has_rand else None}

    enc = f"{cfg.encoder}-{cfg.decoder}"
    entries = [{"name": "Week 1 scratch U-Net (UNetBN, benchmark12)", "params_m": params["week1"],
                "val": week1_val, "test": week1_test}]
    if has_rand:
        entries.append({"name": f"Week 2 {enc}, random init (ablation)", "params_m": params["random"],
                        "val": val_by_model["random"], "test": test_by_model["random"]})
    entries.append({"name": f"Week 2 {enc}, ImageNet-pretrained ({selected})", "params_m": params["pretrained"],
                    "val": val_by_model["pretrained"], "test": test_by_model["pretrained"]})
    model_comparison, numeric = an.build_model_comparison(entries)
    model_comparison.to_csv(dirs["results"] / "model_comparison.csv", index=False)
    numeric.to_csv(dirs["results"] / "model_comparison_numeric.csv", index=False)

    val_frames = {WEEK1_LABEL: week1_val, PRETRAINED_LABEL: val_by_model["pretrained"]}
    if has_rand:
        val_frames[RANDOM_LABEL] = val_by_model["random"]
    paired_val = an.paired_seed_table(val_frames)
    paired_val.to_csv(dirs["results"] / "paired_seed_val_iou.csv", index=False)

    paired_test = None
    test_frames = {PRETRAINED_LABEL: test_by_model["pretrained"]}
    if week1_test is not None:
        test_frames = {WEEK1_LABEL: week1_test, **test_frames}
    if has_rand:
        test_frames[RANDOM_LABEL] = test_by_model["random"]
    if len(test_frames) > 1:
        paired_test = an.paired_seed_table(test_frames)
        paired_test.to_csv(dirs["results"] / "paired_seed_test_iou.csv", index=False)

    sel_counts = counts_df[(counts_df["Strategy"] == selected) & (counts_df["Split"] == "test")]
    coverage_test = an.coverage_group_metrics(sel_counts, split_df)
    coverage_test.to_csv(dirs["results"] / "coverage_group_test_selected.csv", index=False)

    res = {"config": cfg.to_dict(), "selected": selected, "runs_val": runs, "runs_test": test_runs,
           "histories": histories, "strategy_summary": strategy_summary, "strategy_table": strategy_table,
           "val_by_model": val_by_model, "test_by_model": test_by_model, "params": params,
           "model_comparison": model_comparison, "model_comparison_numeric": numeric,
           "paired_val_iou": paired_val, "paired_test_iou": paired_test, "coverage_test": coverage_test}

    # ---- 5. figures + markdown ------------------------------------------
    week1_mean = an.mean_std(week1_val["IoU"])[0]
    fig = dirs["figures"]
    rp.plot_validation_curves(histories, selected, week1_mean, fig / "validation_iou_curves.png", show_figures)
    rp.plot_loss_curves(histories, selected, fig / "training_validation_loss_curves.png", show_figures)
    for split in ("Val", "Test"):
        rp.plot_metric_bars(numeric, split, fig / f"{split.lower()}_metric_comparison.png", show_figures)

    q_seed = cfg.seeds[0]
    q_model = build_model(cfg.encoder, cfg.decoder, INIT_RANDOM, cfg.in_channels).to(device)
    load_checkpoint(dirs["checkpoints"] / f"{run_name(cfg, selected, q_seed)}.pth", q_model, device)
    rp.plot_qualitative(q_model, device, datasets["test"], test_df, image_map,
                        fig / "qualitative_test_predictions.png", show_figures, threshold=cfg.threshold)
    del q_model
    _free_memory()

    markdown = rp.build_results_markdown(res)
    (dirs["root"] / "RESULTS.md").write_text(markdown, encoding="utf-8")
    if cfg.readme_path:
        rp.inject_into_readme(cfg.readme_path, markdown)
    res["markdown"] = markdown
    print(f"\nDone. Results written to {dirs['root']}")
    return res
