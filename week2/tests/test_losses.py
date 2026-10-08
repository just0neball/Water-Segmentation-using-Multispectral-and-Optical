import pytest

torch = pytest.importorskip("torch")

from src.losses import BCEDiceLoss  # noqa: E402


def _batch():
    targets = torch.zeros(2, 1, 8, 8)
    targets[:, :, :4] = 1.0
    good = (targets * 20.0) - 10.0                    # confident, correct logits
    bad = -good
    return good, bad, targets


@pytest.mark.parametrize("mode", ["batch", "per_image"])
def test_loss_is_low_for_correct_and_high_for_wrong(mode):
    good, bad, targets = _batch()
    loss = BCEDiceLoss(dice_mode=mode)
    assert loss(good, targets).item() < 0.05
    assert loss(bad, targets).item() > 1.0


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        BCEDiceLoss(dice_mode="nope")
