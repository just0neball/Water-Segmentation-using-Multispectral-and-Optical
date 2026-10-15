import pytest

from src.metrics import segmentation_metrics_from_counts


def test_known_counts():
    m = segmentation_metrics_from_counts(tp=8, fp=2, fn=2, tn=88)
    assert m["Precision"] == pytest.approx(0.8, abs=1e-6)
    assert m["Recall"] == pytest.approx(0.8, abs=1e-6)
    assert m["F1"] == pytest.approx(0.8, abs=1e-6)
    assert m["IoU"] == pytest.approx(8 / 12, abs=1e-6)


def test_perfect_prediction():
    m = segmentation_metrics_from_counts(tp=10, fp=0, fn=0, tn=10)
    assert all(v == pytest.approx(1.0, abs=1e-6) for v in m.values())


def test_no_water_predicted_gives_zero_not_nan():
    m = segmentation_metrics_from_counts(tp=0, fp=0, fn=5, tn=95)
    assert m["IoU"] == 0.0 and m["Recall"] == 0.0


def test_confusion_counts_from_logits_torch():
    torch = pytest.importorskip("torch")
    from src.metrics import confusion_counts_from_logits, per_sample_counts_from_logits

    logits = torch.tensor([[[[10.0, -10.0], [10.0, -10.0]]]])
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    assert confusion_counts_from_logits(logits, targets) == (1, 1, 1, 1)
    assert per_sample_counts_from_logits(logits, targets) == [[1, 1, 1, 1]]
