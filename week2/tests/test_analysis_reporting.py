"""Tests for the pure-pandas parts (no torch needed)."""

import numpy as np
import pandas as pd
import pytest

from src import analysis as an
from src import reporting as rp


def _runs(iou_values, seeds=(42, 123, 2026)):
    return pd.DataFrame({"Seed": list(seeds), "IoU": iou_values, "F1": 0.85, "Precision": 0.88, "Recall": 0.82})


def test_mean_std_and_formatting():
    assert an.mean_std([0.7, 0.8, 0.9])[0] == pytest.approx(0.8)
    assert an.fmt_mean_std([0.7, 0.8, 0.9]) == "0.8000 ± 0.1000"


def test_paired_seed_table_deltas():
    t = an.paired_seed_table({"A": _runs([0.70, 0.71, 0.72]), "B": _runs([0.75, 0.74, 0.76])})
    assert list(t["Δ B − A"].round(2)) == [0.05, 0.03, 0.04]


def test_week1_baseline_loader(tmp_path):
    raw = pd.DataFrame({
        "Architecture": ["UNetBN"] * 4, "Experiment": ["benchmark12"] * 3 + ["E0"], "Seed": [42, 123, 2026, 42],
        "In_Channels": [12, 12, 12, 7], "Val_Global_IoU": [0.75, 0.76, 0.77, 0.5],
        "Val_F1": 0.86, "Val_Precision": 0.9, "Val_Recall": 0.83})
    path = tmp_path / "w1.csv"
    raw.to_csv(path, index=False)
    out = an.load_week1_baseline(path)
    assert list(out["Seed"]) == [42, 123, 2026] and list(out.columns) == ["Seed", *an.METRICS]


def test_week1_test_loader_accepts_alternative_column_names(tmp_path):
    pd.DataFrame({"Seed": [42, 123], "Test_Global_IoU": [0.68, 0.69], "Test_F1": 0.8,
                  "Test_Precision": 0.85, "Test_Recall": 0.78}).to_csv(tmp_path / "t.csv", index=False)
    out = an.load_week1_test_baseline(tmp_path / "t.csv")
    assert out["IoU"].tolist() == [0.68, 0.69]


def test_coverage_group_metrics_handles_no_water_group():
    split = pd.DataFrame({"Sample_ID": [1, 2, 3], "Coverage_Group": ["No Water", "Low", "Low"],
                          "Water_Percentage": [0.0, 10.0, 12.0]})
    counts = pd.DataFrame({"Sample_ID": [1, 2, 3], "TP": [0, 80, 90], "FP": [5, 10, 5],
                           "FN": [0, 10, 10], "TN": [995, 900, 895]})
    out = an.coverage_group_metrics(counts, split)
    assert out["Coverage_Group"].tolist() == ["No Water", "Low"]
    assert np.isnan(out.loc[0, "IoU"]) and out.loc[0, "FP_Pixel_Rate"] == pytest.approx(0.005)
    assert out.loc[1, "IoU"] == pytest.approx(170 / (170 + 15 + 20), abs=1e-6)


def test_mean_image_iou_skips_images_without_water():
    counts = pd.DataFrame({"TP": [0, 50], "FP": [9, 10], "FN": [0, 40]})
    assert an.mean_image_iou_water_only(counts) == pytest.approx(50 / 100)


def test_readme_injection_replaces_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"# T\n\n{rp.README_START}\nold\n{rp.README_END}\n\nfooter\n", encoding="utf-8")
    rp.inject_into_readme(readme, "NEW TABLE")
    text = readme.read_text(encoding="utf-8")
    assert "NEW TABLE" in text and "old" not in text and text.rstrip().endswith("footer")
    rp.inject_into_readme(readme, "NEWER")
    assert readme.read_text(encoding="utf-8").count(rp.README_START) == 1


def test_findings_are_hedged_when_ablation_gap_is_within_noise():
    w1, pre, rnd = _runs([0.754, 0.752, 0.756]), _runs([0.80, 0.77, 0.78]), _runs([0.79, 0.76, 0.785])
    res = {
        "selected": "mean",
        "val_by_model": {"week1": w1, "pretrained": pre, "random": rnd},
        "test_by_model": {"week1": None, "pretrained": pre, "random": rnd},
        "paired_val_iou": an.paired_seed_table({"Week 1 scratch": w1, "Pretrained": pre, "Random init": rnd}),
        "strategy_summary": pd.DataFrame({"Strategy": ["mean", "rgb_zero_extra"],
                                          "Val_IoU_mean": [0.783, 0.778], "Val_IoU_std": [0.013, 0.012]}),
        "params": {"week1": None, "pretrained": 24.4, "random": 24.4},
    }
    text = rp.make_findings(res)
    assert "cannot be attributed" in text
    assert "not supplied" in text                       # no Week 1 test CSV
    assert "not statistically meaningful" in text       # init-strategy gap < seed std
    assert "3/3 seeds" in text


def test_df_to_markdown_formats_nan_and_floats():
    md = rp.df_to_markdown(pd.DataFrame({"a": [0.123456, np.nan], "b": ["x", "y"]}))
    assert "0.1235" in md and "n/a" in md


def test_metric_bar_plot_writes_file(tmp_path):
    numeric = pd.DataFrame([{"Model": m, "Split": "Val", "Metric": k, "Mean": 0.8, "Std": 0.01, "N_seeds": 3}
                            for m in ("A", "B") for k in an.METRICS])
    out = tmp_path / "bars.png"
    rp.plot_metric_bars(numeric, "Val", out)
    assert out.exists() and out.stat().st_size > 0


def test_full_results_markdown_builds_from_synthetic_results():
    w1, pre, rnd = _runs([0.754, 0.752, 0.756]), _runs([0.80, 0.77, 0.78]), _runs([0.72, 0.73, 0.74])
    entries = [{"name": "W1", "params_m": None, "val": w1, "test": None},
               {"name": "Rand", "params_m": 24.4, "val": rnd, "test": rnd},
               {"name": "Pre", "params_m": 24.4, "val": pre, "test": pre}]
    comparison, _ = an.build_model_comparison(entries)
    res = {
        "config": {"encoder": "resnet34", "decoder": "unet", "seeds": (42, 123, 2026)},
        "selected": "mean",
        "val_by_model": {"week1": w1, "pretrained": pre, "random": rnd},
        "test_by_model": {"week1": None, "pretrained": pre, "random": rnd},
        "paired_val_iou": an.paired_seed_table({"Week 1 scratch": w1, "Pretrained": pre, "Random init": rnd}),
        "paired_test_iou": an.paired_seed_table({"Pretrained": pre, "Random init": rnd}),
        "strategy_summary": pd.DataFrame({"Strategy": ["mean", "rgb_zero_extra"],
                                          "Val_IoU_mean": [0.783, 0.70], "Val_IoU_std": [0.013, 0.01]}),
        "strategy_table": pd.DataFrame({"Strategy": ["mean"], "Mean ± Std": ["0.78 ± 0.01"]}),
        "params": {"week1": 7.8, "pretrained": 24.4, "random": 24.4},
        "model_comparison": comparison,
        "coverage_test": pd.DataFrame({"Coverage_Group": ["Low"], "N_images": [3], "IoU": [0.7]}),
    }
    md = rp.build_results_markdown(res)
    for needle in ("### Model comparison", "### Limitations", "Ablation", "7.80 M", "ImageNet weights contribute"):
        assert needle in md
