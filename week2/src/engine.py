"""Training / evaluation loops and checkpoint helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .metrics import (confusion_counts_from_logits, per_sample_counts_from_logits,
                      segmentation_metrics_from_counts)
from .utils import seed_everything


def _autocast(device):
    return torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"))


def build_optimizer(model, encoder_lr: float, decoder_lr: float, weight_decay: float):
    """Differential learning rates: small for the pretrained encoder, larger for the decoder."""
    decoder_params = list(model.decoder.parameters()) + list(model.segmentation_head.parameters())
    return torch.optim.AdamW(
        [{"params": list(model.encoder.parameters()), "lr": encoder_lr},
         {"params": decoder_params, "lr": decoder_lr}],
        weight_decay=weight_decay,
    )


def train_one_epoch(model, loader, optimizer, criterion, device, scaler) -> float:
    model.train()
    running_loss, total = 0.0, 0
    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)
    return running_loss / max(total, 1)


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, threshold: float = 0.5) -> dict:
    """Global (pooled-pixel) IoU / F1 / Precision / Recall + mean loss for one split."""
    model.eval()
    running_loss, total = 0.0, 0
    tp = fp = fn = tn = 0
    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with _autocast(device):
            logits = model(images)
            loss = criterion(logits, masks)
        running_loss += loss.item() * images.size(0)
        total += images.size(0)
        a, b, c, d = confusion_counts_from_logits(logits, masks, threshold)
        tp, fp, fn, tn = tp + a, fp + b, fn + c, tn + d

    metrics = segmentation_metrics_from_counts(tp, fp, fn, tn)
    metrics["Loss"] = running_loss / max(total, 1)
    return metrics


@torch.no_grad()
def predict_sample_counts(model, loader, device, threshold: float = 0.5) -> pd.DataFrame:
    """Per-image confusion counts (Sample_ID, TP, FP, FN, TN) for error analysis."""
    model.eval()
    rows = []
    for images, masks, ids in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with _autocast(device):
            logits = model(images)
        for sid, (tp, fp, fn, tn) in zip(ids, per_sample_counts_from_logits(logits, masks, threshold)):
            rows.append({"Sample_ID": int(sid), "TP": tp, "FP": fp, "FN": fn, "TN": tn})
    return pd.DataFrame(rows)


@torch.no_grad()
def predict_probabilities(model, images: torch.Tensor, device) -> np.ndarray:
    """Sigmoid probabilities (N, H, W) for a batch of preprocessed images (N, 12, H, W)."""
    model.eval()
    with _autocast(device):
        logits = model(images.to(device))
    return torch.sigmoid(logits.float())[:, 0].cpu().numpy()


def fit_model(model, train_loader, val_loader, criterion, device, *, seed: int,
              max_epochs: int = 100, patience: int = 15,
              encoder_lr: float = 1e-4, decoder_lr: float = 1e-3, weight_decay: float = 1e-4,
              threshold: float = 0.5, min_delta: float = 1e-5,
              scheduler_patience: int = 4, scheduler_factor: float = 0.5, min_lr: float = 1e-6,
              log_every: int = 5, verbose: bool = True):
    """Fine-tune with early stopping on validation IoU; restore the best weights.

    Returns ``(model, history_df, best_epoch, best_val_iou)``.
    """
    seed_everything(seed)
    model = model.to(device)
    optimizer = build_optimizer(model, encoder_lr, decoder_lr, weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=scheduler_factor, patience=scheduler_patience, min_lr=min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_iou, best_epoch, stale, best_state, history = -np.inf, 0, 0, None, []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val = evaluate_model(model, val_loader, criterion, device, threshold)
        scheduler.step(val["IoU"])

        history.append({
            "Epoch": epoch, "Train_Loss": train_loss, "Val_Loss": val["Loss"],
            "Val_IoU": val["IoU"], "Val_F1": val["F1"],
            "Val_Precision": val["Precision"], "Val_Recall": val["Recall"],
            "Encoder_LR": optimizer.param_groups[0]["lr"],
            "Decoder_LR": optimizer.param_groups[1]["lr"],
        })

        improved = val["IoU"] > best_iou + min_delta
        if improved:
            best_iou, best_epoch, stale = val["IoU"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1

        if verbose and (improved or epoch == 1 or epoch % log_every == 0):
            print(f"  epoch {epoch:03d} | train loss {train_loss:.4f} | val loss {val['Loss']:.4f} | "
                  f"IoU {val['IoU']:.4f} | F1 {val['F1']:.4f} | "
                  f"P {val['Precision']:.4f} | R {val['Recall']:.4f}{'  *' if improved else ''}")

        if stale >= patience:
            if verbose:
                print(f"  early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint was recorded during training.")
    model.load_state_dict(best_state)
    if verbose:
        print(f"  best epoch {best_epoch} | best val IoU {best_iou:.4f}")
    return model, pd.DataFrame(history), best_epoch, best_iou


def save_checkpoint(path, model, meta: dict) -> None:
    torch.save({"model_state_dict": model.state_dict(), **meta}, path)


def load_checkpoint(path, model, device) -> dict:
    """Load weights into ``model`` and return the stored metadata."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    return {k: v for k, v in ckpt.items() if k != "model_state_dict"}


def read_checkpoint_meta(path) -> dict:
    """Read only the metadata of a checkpoint (no model needed)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v for k, v in ckpt.items() if k != "model_state_dict"}
