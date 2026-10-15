"""Pretrained segmentation model (segmentation-models-pytorch) for 12-channel input.

ImageNet encoders expect 3 channels, so the first convolution is replaced by a
12-channel convolution whose weights are derived from the pretrained RGB filters:

* ``mean``            - average the RGB filters, repeat them over all 12 channels and
                        scale by 3/12 (keeps the activation magnitude roughly unchanged).
* ``rgb_zero_extra``  - copy the R/G/B filters onto the dataset's Red/Green/Blue bands
                        (indices 3/2/1) and zero-initialise the other 9 channels.
* ``random_init``     - ablation: same architecture, *no* ImageNet weights at all.
"""

from __future__ import annotations

import copy

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from .preprocessing import RGB_CHANNEL_INDICES

INIT_MEAN = "mean"
INIT_RGB_ZERO = "rgb_zero_extra"
INIT_RANDOM = "random_init"
PRETRAINED_STRATEGIES = (INIT_MEAN, INIT_RGB_ZERO)

DECODERS = {"unet": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus}


def adapt_first_conv_weights(weight: torch.Tensor, in_channels: int = 12,
                             strategy: str = INIT_MEAN,
                             rgb_target_channels=RGB_CHANNEL_INDICES) -> torch.Tensor:
    """Turn pretrained (out, 3, k, k) weights into (out, in_channels, k, k).

    ImageNet filter order is R, G, B; ``rgb_target_channels`` gives the dataset
    channel index of each (default: Red=3, Green=2, Blue=1).
    """
    if weight.ndim != 4 or weight.shape[1] != 3:
        raise ValueError(f"Expected pretrained weights of shape (out, 3, k, k), got {tuple(weight.shape)}")

    w = weight.detach().clone()
    if strategy == INIT_MEAN:
        return w.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1) * (3.0 / in_channels)

    if strategy == INIT_RGB_ZERO:
        new = torch.zeros(w.shape[0], in_channels, *w.shape[2:], dtype=w.dtype, device=w.device)
        for src, dst in enumerate(rgb_target_channels):    # src: 0=R 1=G 2=B
            new[:, dst] = w[:, src]
        return new

    raise ValueError(f"Unknown strategy '{strategy}'. Use '{INIT_MEAN}' or '{INIT_RGB_ZERO}'.")


def find_first_conv(encoder: nn.Module):
    """Locate the stem convolution (first Conv2d with 3 input channels).

    Works for ResNet (`conv1`) and EfficientNet (`_conv_stem`) encoders alike.
    """
    for name, module in encoder.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            return name, module
    raise ValueError("No 3-channel stem convolution found in the encoder.")


def adapt_first_conv(encoder: nn.Module, in_channels: int = 12, strategy: str = INIT_MEAN) -> nn.Module:
    """Replace the encoder stem conv by an `in_channels` conv (in place)."""
    name, old_conv = find_first_conv(encoder)
    if old_conv.groups != 1:
        raise ValueError("Grouped stem convolutions are not supported.")

    new_conv = copy.deepcopy(old_conv)          # keeps bias / padding / subclass behaviour
    new_conv.in_channels = in_channels
    new_conv.weight = nn.Parameter(adapt_first_conv_weights(old_conv.weight, in_channels, strategy))

    *path, attr = name.split(".")
    parent = encoder
    for part in path:
        parent = getattr(parent, part)
    setattr(parent, attr, new_conv)

    if hasattr(encoder, "_in_channels"):         # keep smp metadata consistent
        encoder._in_channels = in_channels
    return encoder


def build_model(encoder_name: str = "resnet34", decoder_name: str = "unet",
                init_strategy: str = INIT_MEAN, in_channels: int = 12,
                classes: int = 1) -> nn.Module:
    """Build the segmentation model (outputs raw logits, no activation)."""
    if decoder_name not in DECODERS:
        raise ValueError(f"decoder_name must be one of {sorted(DECODERS)}")
    decoder_cls = DECODERS[decoder_name]

    if init_strategy == INIT_RANDOM:             # ablation: no pretrained weights
        return decoder_cls(encoder_name=encoder_name, encoder_weights=None,
                           in_channels=in_channels, classes=classes, activation=None)

    model = decoder_cls(encoder_name=encoder_name, encoder_weights="imagenet",
                        in_channels=3, classes=classes, activation=None)
    adapt_first_conv(model.encoder, in_channels, init_strategy)
    return model
