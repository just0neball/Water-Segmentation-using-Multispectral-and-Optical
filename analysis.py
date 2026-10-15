"""Tables and statistics: Week 1 baseline loading, comparisons, error analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import segmentation_metrics_from_counts

METRICS = ["IoU", "F1", "Precision", "Recall"]


# --------------------------------------------------------------------------
# Basic statistics
# --------------------------------------------------------------------------


def mean_std(values) -> tuple:
    values = np.asarray(list(values), dtype=float)
    std = values.std(ddof=1) if len(values) > 1 else float("nan")
    return float(values.mean()), float(std)


def fmt_mean_std(values, digits: int = 4) -> str:
    m, s = mean_std(values)
    return f"{m:.{digits}f} ± {s:.{digits}f}" if not np.isnan(s) else f"{m:.{digits}f}"


def summarize_runs(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Mean / std of `<prefix>_<metric>` columns across rows (seeds)."""
    rows = []
    for metric in METRICS:
        m, s = mean_std(df[f"{prefix}_{metric}"])
        rows.append({"Metric": metric, "Mean": m, "Std": s, "Mean ± Std": fmt_mean_std(df[f"{prefix}_{metric}"])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Week 1 baseline
# --------------------------------------------------------------------------


def load_week1_baseline(source, architecture: str = "UNetBN", experiment: str = "benchmark12",
                        in_channels: int = 12) -> pd.DataFrame:
    """Week 1 scratch U-Net validation results -> columns Seed, IoU, F1, Precision, Recall."""
    raw = pd.read_csv(source)
    sel = raw[(raw["Architecture"] == architecture)
              & (raw["Experiment"] == experiment)
              & (raw["In_Channels"] == in_channels)].copy()
    if sel.empty:
        raise ValueError(f"No Week 1 rows for {architecture}/{experiment}/{in_channels}ch in {source}")
    out = sel.rename(columns={"Val_Global_IoU": "IoU", "Val_F1": "F1",
                              "Val_Precision": "Precision", "Val_Recall": "Recall"})
    return out[["Seed"] + METRICS].sort_values("Seed").reset_index(drop=True)


_TEST_COLUMN_CANDIDATES = {
    "IoU": ["Test_Global_IoU", "Test_IoU", "IoU"],
    "F1": ["Test_F1", "F1"],
    "Precision": ["Test_Precision", "Precision"],
    "Recall": ["Test_Recall", "Recall"],
}


def load_week1_test_baseline(source, architecture: str = "UNetBN", experiment: str = "benchmark12"):
    """Optional Week 1 test metrics per seed (CSV needs a Seed column + IoU/F1/Precision/Recall).

    Column names `Test_IoU`/`Test_Global_IoU`/`IoU` (and the F1/Precision/Recall equivalents)
    are all accepted. If `Architecture`/`Experiment` columns exist the rows are filtered.
    """
    raw = pd.read_csv(source)
    if {"Architecture", "Experiment"} <= set(raw.columns):
        filtered = raw[(raw["Architecture"] == architecture) & (raw["Experiment"] == experiment)]
        raw = filtered if not filtered.empty else raw
    out = pd.DataFrame({"Seed": raw["Seed"] if "Seed" in raw.columns else range(len(raw))})
    for metric, candidates in _TEST_COLUMN_CANDIDATES.items():
        col = next((c for c in candidates if c in raw.columns), None)
        if col is None:
            raise ValueError(f"{source}: no column found for {metric} (tried {candidates})")
        out[metric] = raw[col].values
    return out.sort_values("Seed").reset_index(drop=True)


# --------------------------------------------------------------------------
# Comparison tables
# --------------------------------------------------------------------------


def wide_to_plain(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Rename `<prefix>_IoU` -> `IoU` etc. so different sources share column names."""
    return df.rename(columns={f"{prefix}_{m}": m for m in METRICS})[["Seed"] + METRICS]


def build_model_comparison(entries: list) -> tuple:
    """entries: [{name, params_m, val(df|None), test(df|None)}] -> (pretty_df, numeric_df)."""
    pretty_rows, numeric_rows = [], []
    for e in entries:
        row = {"Model": e["name"],
               "Params (M)": f"{e['params_m']:.2f}" if e.get("params_m") else "n/a"}
        for split, key in (("Val", "val"), ("Test", "test")):
            df = e.get(key)
            for metric in METRICS:
                if df is None:
                    row[f"{split} {metric}"] = "n/a"
                else:
                    row[f"{split} {metric}"] = fmt_mean_std(df[metric])
                    m, s = mean_std(df[metric])
                    numeric_rows.append({"Model": e["name"], "Split": split, "Metric": metric,
                                         "Mean": m, "Std": s, "N_seeds": len(df)})
        pretty_rows.append(row)
    return pd.DataFrame(pretty_rows), pd.DataFrame(numeric_rows)


def paired_seed_table(frames: dict, metric: str = "IoU") -> pd.DataFrame:
    """frames: {label: df with Seed + metric}. Inner-join on Seed; adds delta columns vs first."""
    labels = list(frames)
    out = None
    for label in labels:
        part = frames[label][["Seed", metric]].rename(columns={metric: label})
        out = part if out is None else out.merge(part, on="Seed", how="inner")
    out = out.sort_values("Seed").reset_index(drop=True)
    base = labels[0]
    for label in labels[1:]:
        out[f"Δ {label} − {base}"] = out[label] - out[base]
    return out


# --------------------------------------------------------------------------
# Error analysis
# --------------------------------------------------------------------------


def mean_image_iou_water_only(counts_df: pd.DataFrame) -> float:
    """Mean per-image IoU over images that contain water (images without water are skipped).

    NOTE: Week 1 may use a different convention for its `Val_Mean_Image_IoU`; only
    compare this number between Week 2 models.
    """
    sub = counts_df[(counts_df["TP"] + counts_df["FN"]) > 0]
    if sub.empty:
        return float("nan")
    return float((sub["TP"] / (sub["TP"] + sub["FP"] + sub["FN"])).mean())


def coverage_group_metrics(counts_df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    """Pooled IoU/F1/P/R per water-coverage group (+ false-positive pixel rate)."""
    merged = counts_df.merge(split_df[["Sample_ID", "Coverage_Group", "Water_Percentage"]], on="Sample_ID")
    order = merged.groupby("Coverage_Group")["Water_Percentage"].min().sort_values().index
    rows = []
    for group in order:
        g = merged[merged["Coverage_Group"] == group]
        tp, fp, fn, tn = (int(g[c].sum()) for c in ("TP", "FP", "FN", "TN"))
        row = {"Coverage_Group": group, "N_images": len(g), "FP_Pixel_Rate": fp / max(tp + fp + fn + tn, 1)}
        if tp + fn == 0:     # no water in ground truth -> IoU undefined, only the FP rate matters
            row.update({m: float("nan") for m in METRICS})
        else:
            row.update(segmentation_metrics_from_counts(tp, fp, fn, tn))
        rows.append(row)
    return pd.DataFrame(rows)
