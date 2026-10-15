"""Figures, markdown tables, auto-generated findings and README injection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt  # noqa: E402

from .analysis import METRICS, mean_std        # noqa: E402
from .preprocessing import make_rgb_composite  # noqa: E402

README_START = "<!-- RESULTS:START -->"
README_END = "<!-- RESULTS:END -->"


# --------------------------------------------------------------------------
# Markdown helpers
# --------------------------------------------------------------------------


def df_to_markdown(df: pd.DataFrame, digits: int = 4) -> str:
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "n/a" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def inject_into_readme(readme_path, markdown: str) -> None:
    """Replace the block between the RESULTS markers (or append it if the markers are absent)."""
    readme_path = Path(readme_path)
    block = f"{README_START}\n{markdown.strip()}\n{README_END}"
    text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    if README_START in text and README_END in text:
        head, rest = text.split(README_START, 1)
        _, tail = rest.split(README_END, 1)
        text = head + block + tail
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    readme_path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _finish(fig, out_path, show):
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_validation_curves(histories: dict, selected: str, week1_mean_iou, out_path=None, show=False):
    """Validation IoU vs epoch. histories: {(strategy, seed): history_df}."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for (strategy, seed), h in histories.items():
        if strategy == selected:
            ax.plot(h["Epoch"], h["Val_IoU"], color=colors[seed % len(colors)],
                    label=f"{selected} - seed {seed}")
        elif strategy == "random_init":
            ax.plot(h["Epoch"], h["Val_IoU"], color=colors[seed % len(colors)], linestyle=":",
                    alpha=0.8, label=f"random init - seed {seed}")
    if week1_mean_iou is not None:
        ax.axhline(week1_mean_iou, linestyle="--", color="black", linewidth=2,
                   label=f"Week 1 scratch U-Net mean ({week1_mean_iou:.4f})")
    ax.set(xlabel="Epoch", ylabel="Validation IoU", title="Validation IoU per epoch")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _finish(fig, out_path, show)


def plot_loss_curves(histories: dict, selected: str, out_path=None, show=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for (strategy, seed), h in histories.items():
        if strategy != selected:
            continue
        c = colors[seed % len(colors)]
        ax.plot(h["Epoch"], h["Train_Loss"], color=c, label=f"train - seed {seed}")
        ax.plot(h["Epoch"], h["Val_Loss"], color=c, linestyle="--", label=f"val - seed {seed}")
    ax.set(xlabel="Epoch", ylabel="BCE + Dice loss", title=f"Training / validation loss ({selected})")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _finish(fig, out_path, show)


def plot_metric_bars(numeric_df: pd.DataFrame, split: str, out_path=None, show=False):
    sub = numeric_df[numeric_df["Split"] == split]
    if sub.empty:
        return
    models = list(dict.fromkeys(sub["Model"]))
    x = np.arange(len(METRICS))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, model in enumerate(models):
        part = sub[sub["Model"] == model].set_index("Metric").loc[METRICS]
        yerr = part["Std"].fillna(0).values
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * width, part["Mean"].values, width,
                      yerr=yerr, capsize=3, label=model)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, METRICS)
    ax.set(ylim=(0, 1.08), ylabel=f"{split} score", title=f"{split} metrics (mean ± std over seeds)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    _finish(fig, out_path, show)


def plot_qualitative(model, device, test_dataset, test_df, image_map, out_path=None, show=False,
                     quantiles=(0.05, 0.35, 0.65, 0.95), threshold=0.5):
    """RGB | ground truth | prediction | error map for samples spread over water coverage."""
    import torch
    from .data import load_raw_image
    from .engine import predict_probabilities

    ordered = test_df.sort_values("Water_Percentage").reset_index(drop=True)
    picks = ordered.iloc[[int(round((len(ordered) - 1) * q)) for q in quantiles]]

    fig, axes = plt.subplots(len(picks), 4, figsize=(14, 3.6 * len(picks)))
    for r, (_, row) in enumerate(picks.iterrows()):
        sid = int(row["Sample_ID"])
        image, gt = test_dataset.get_processed(sid)
        prob = predict_probabilities(model, torch.from_numpy(image).unsqueeze(0).float(), device)[0]
        pred = prob >= threshold
        error = np.zeros((*gt.shape, 3), dtype=np.float32)
        error[(pred == 1) & (gt == 1)] = (0.2, 0.7, 0.2)     # TP green
        error[(pred == 1) & (gt == 0)] = (0.9, 0.2, 0.2)     # FP red
        error[(pred == 0) & (gt == 1)] = (0.2, 0.3, 0.9)     # FN blue

        panels = [(make_rgb_composite(load_raw_image(image_map[str(sid)])),
                   f"RGB - ID {sid} ({row['Water_Percentage']:.1f}% water)", None),
                  (gt, "Ground truth", "gray"), (pred.astype(np.float32), "Prediction", "gray"),
                  (error, "TP green / FP red / FN blue", None)]
        for c, (img, title, cmap) in enumerate(panels):
            axes[r, c].imshow(img, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
            axes[r, c].set_title(title, fontsize=9)
            axes[r, c].axis("off")
    _finish(fig, out_path, show)


# --------------------------------------------------------------------------
# Auto-generated findings (wording depends on what the numbers actually show)
# --------------------------------------------------------------------------


def make_findings(res: dict) -> str:
    """Findings paragraph generated from the computed results (hedged where the data is weak)."""
    val = res["val_by_model"]
    out = []

    pre_m, pre_s = mean_std(val["pretrained"]["IoU"])
    w1_m, w1_s = mean_std(val["week1"]["IoU"])
    paired = res["paired_val_iou"]
    delta_col = "Δ Pretrained − Week 1 scratch"
    k = int((paired[delta_col] > 0).sum())
    out.append(f"- **Pretrained vs Week 1 scratch (validation):** mean IoU {pre_m:.4f} ± {pre_s:.4f} vs "
               f"{w1_m:.4f} ± {w1_s:.4f} (Δ = {pre_m - w1_m:+.4f}); higher in {k}/{len(paired)} seeds.")

    for metric in ("Precision", "Recall", "F1"):
        d = mean_std(val["pretrained"][metric])[0] - mean_std(val["week1"][metric])[0]
        out.append(f"  - {metric}: {d:+.4f}")

    test = res["test_by_model"]
    if test.get("week1") is not None:
        t_pre, t_w1 = mean_std(test["pretrained"]["IoU"]), mean_std(test["week1"]["IoU"])
        out.append(f"- **Held-out test:** mean IoU {t_pre[0]:.4f} (pretrained) vs {t_w1[0]:.4f} (Week 1 scratch), "
                   f"Δ = {t_pre[0] - t_w1[0]:+.4f}. The test set was not used for any selection, so this is the "
                   f"less optimistic comparison (validation IoU was used for early stopping in both weeks).")
    else:
        out.append("- **Held-out test:** Week 1 test metrics were not supplied (`--week1-test-csv`), so only the "
                   "Week 2 test results are reported; no test-set comparison with Week 1 is made.")

    if test.get("random") is not None and val.get("random") is not None:
        r_m, r_s = mean_std(val["random"]["IoU"])
        d, noise = pre_m - r_m, max(pre_s, r_s)
        if d > noise:
            verdict = ("the gap exceeds the seed-to-seed spread, which suggests the ImageNet weights contribute "
                       "beyond architecture and capacity (still only 3 seeds)")
        elif d > 0:
            verdict = ("the gap is within the seed-to-seed spread, so the gain cannot be attributed to the "
                       "ImageNet weights with confidence")
        else:
            verdict = ("random initialisation does at least as well, so the gain over Week 1 comes from the "
                       "architecture/capacity (deeper ResNet34 encoder, more parameters), not from pretraining")
        out.append(f"- **Ablation (same ResNet34 architecture, random init):** val IoU {r_m:.4f} ± {r_s:.4f} vs "
                   f"{pre_m:.4f} ± {pre_s:.4f} pretrained (Δ = {d:+.4f}); {verdict}.")
        if r_m > w1_m:
            out.append(f"  - The random-init ResNet34-U-Net already beats the Week 1 model by {r_m - w1_m:+.4f} IoU, "
                       "so part of the improvement is architectural.")

    strat = res["strategy_summary"]
    if len(strat) > 1:
        best, other = strat.iloc[0], strat.iloc[1]
        gap = best["Val_IoU_mean"] - other["Val_IoU_mean"]
        noise = max(best["Val_IoU_std"], other["Val_IoU_std"])
        note = "within seed variability, so the choice is not statistically meaningful" if gap <= noise \
            else "larger than the seed spread"
        out.append(f"- **First-conv initialisation:** `{best['Strategy']}` ({best['Val_IoU_mean']:.4f}) vs "
                   f"`{other['Strategy']}` ({other['Val_IoU_mean']:.4f}) - gap {gap:+.4f}, {note}.")

    p = res["params"]
    if p.get("week1"):
        out.append(f"- **Capacity:** Week 2 model {p['pretrained']:.2f} M parameters vs Week 1 {p['week1']:.2f} M.")
    else:
        out.append(f"- **Capacity:** Week 2 model has {p['pretrained']:.2f} M parameters (Week 1 count not supplied).")
    return "\n".join(out)


LIMITATIONS = """\
- Only 3 seeds and a single 214/46/46 split; validation has just 46 patches, so small differences are noisy.
- Validation IoU drives early stopping (and the init-strategy choice), so validation numbers are optimistic for
  every model; the held-out test numbers are the unbiased estimate.
- Hyper-parameters were reused from Week 1 and not tuned for the pretrained model.
- The 12 channels include QA, ESA WorldCover and Water Occurrence, which can act as shortcut features
  (see the Week 1 shortcut audit). Results therefore describe this 12-channel setting, not pure spectral learning.
- ImageNet RGB statistics differ strongly from NIR/SWIR/DEM/categorical channels; the benefit of the pretrained
  *first* layer is limited, most transferable value is expected in deeper layers (a hypothesis, not tested here)."""


def build_results_markdown(res: dict) -> str:
    parts = ["## Results", "",
             f"Encoder `{res['config']['encoder']}` + `{res['config']['decoder']}` decoder, "
             f"seeds {list(res['config']['seeds'])}, selected first-conv init: **`{res['selected']}`**.", "",
             "### Model comparison (mean ± std over seeds)", "",
             df_to_markdown(res["model_comparison"]), "",
             "### Per-seed validation IoU", "", df_to_markdown(res["paired_val_iou"]), ""]
    if res.get("paired_test_iou") is not None:
        parts += ["### Per-seed test IoU", "", df_to_markdown(res["paired_test_iou"]), ""]
    parts += ["### First-layer initialisation strategies (validation IoU)", "",
              df_to_markdown(res["strategy_table"]), "",
              "### Error analysis by water coverage (selected model, held-out test; counts pooled over seeds, `N_images` counts each image once per seed)", "",
              df_to_markdown(res["coverage_test"]), "",
              "### Findings (auto-generated from the numbers above)", "", make_findings(res), "",
              "### Limitations", "", LIMITATIONS]
    return "\n".join(parts)
