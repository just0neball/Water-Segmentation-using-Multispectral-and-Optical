import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
pytest.importorskip("segmentation_models_pytorch")

from src.model import (INIT_MEAN, INIT_RANDOM, INIT_RGB_ZERO, adapt_first_conv,  # noqa: E402
                       adapt_first_conv_weights, build_model)


def test_mean_strategy_shape_and_scale():
    w = torch.randn(64, 3, 7, 7)
    out = adapt_first_conv_weights(w, 12, INIT_MEAN)
    assert out.shape == (64, 12, 7, 7)
    assert torch.allclose(out.sum(dim=1), w.mean(dim=1) * 3.0, atol=1e-5)   # response preserved
    assert torch.allclose(out[:, 0], out[:, 11])                             # all channels identical


def test_rgb_zero_extra_maps_rgb_to_dataset_bands_and_zeroes_rest():
    w = torch.randn(64, 3, 7, 7)
    out = adapt_first_conv_weights(w, 12, INIT_RGB_ZERO)
    assert torch.equal(out[:, 3], w[:, 0])      # ImageNet R -> dataset Red   (index 3)
    assert torch.equal(out[:, 2], w[:, 1])      # ImageNet G -> dataset Green (index 2)
    assert torch.equal(out[:, 1], w[:, 2])      # ImageNet B -> dataset Blue  (index 1)
    others = [c for c in range(12) if c not in (1, 2, 3)]
    assert torch.count_nonzero(out[:, others]) == 0


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        adapt_first_conv_weights(torch.randn(8, 4, 3, 3), 12, INIT_MEAN)
    with pytest.raises(ValueError):
        adapt_first_conv_weights(torch.randn(8, 3, 3, 3), 12, "bogus")


def test_adapt_first_conv_replaces_stem_in_place():
    encoder = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1))
    adapt_first_conv(encoder, 12, INIT_MEAN)
    assert encoder[0].in_channels == 12 and encoder[0].weight.shape == (8, 12, 3, 3)
    assert encoder(torch.randn(1, 12, 16, 16)).shape == (1, 8, 16, 16)


def test_random_init_model_forward_12_channels():
    model = build_model("resnet18", "unet", INIT_RANDOM, in_channels=12).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 12, 128, 128))
    assert out.shape == (2, 1, 128, 128)
    assert torch.isfinite(out).all()
