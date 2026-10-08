"""Pixel-level segmentation metrics for the *water* (positive) class.

Metrics are computed from confusion counts accumulated over a whole split,
not by averaging per-batch scores. torch is imported lazily so the formulas
stay importable (and unit-testable) without it.
"""

from __future__ import annotations

METRIC_EPS = 1e-8


def segmentation_metrics_from_counts(tp, fp, fn, tn=0, eps: float = METRIC_EPS) -> dict:
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return {"IoU": iou, "F1": f1, "Precision": precision, "Recall": recall}


def confusion_counts_from_logits(logits, targets, threshold: float = 0.5):
    """Return (tp, fp, fn, tn) as python ints with a single GPU->CPU sync."""
    import torch

    preds = logits.sigmoid() >= threshold
    target = targets >= 0.5
    counts = torch.stack([
        (preds & target).sum(),
        (preds & ~target).sum(),
        (~preds & target).sum(),
        (~preds & ~target).sum(),
    ])
    tp, fp, fn, tn = (int(v) for v in counts.tolist())
    return tp, fp, fn, tn


def per_sample_counts_from_logits(logits, targets, threshold: float = 0.5):
    """Per-image (tp, fp, fn, tn) -> list of 4-int lists, one per image in the batch."""
    import torch

    preds = (logits.sigmoid() >= threshold).flatten(1)
    target = (targets >= 0.5).flatten(1)
    counts = torch.stack([
        (preds & target).sum(1),
        (preds & ~target).sum(1),
        (~preds & target).sum(1),
        (~preds & ~target).sum(1),
    ], dim=1)
    return counts.tolist()
