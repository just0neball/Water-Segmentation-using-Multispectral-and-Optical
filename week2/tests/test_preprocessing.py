import numpy as np
import pytest

from src.preprocessing import IMAGE_SHAPE, make_rgb_composite, preprocess_12ch_benchmark

STATS = {
    "spectral_clip_low": [0.0] * 7, "spectral_clip_high": [1000.0] * 7,
    "spectral_mean": [500.0] * 7, "spectral_std": [100.0] * 7,
    "qa_mean": 0.0, "qa_std": 1.0, "merit_mean": 0.0, "merit_std": 1.0,
    "copernicus_mean": 0.0, "copernicus_std": 1.0, "worldcover_mean": 0.0, "worldcover_std": 1.0,
    "water_occ_mean": 0.0, "water_occ_std": 1.0,
}


def _image(value=10.0):
    return np.full(IMAGE_SHAPE, value, dtype=np.float32)


def test_output_shape_dtype_and_finite():
    out = preprocess_12ch_benchmark(_image(), STATS)
    assert out.shape == (12, 128, 128)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_spectral_bands_are_clipped_then_standardised():
    img = _image()
    img[:7] = 5000.0                                    # far above the clip bound (1000)
    out = preprocess_12ch_benchmark(img, STATS)
    assert np.allclose(out[:7], (1000.0 - 500.0) / 100.0, atol=1e-4)


def test_missing_merit_values_are_filled_from_copernicus():
    img = _image()
    img[8, :10] = -9999.0
    img[9, :] = 42.0
    out = preprocess_12ch_benchmark(img, STATS)
    assert np.allclose(out[8, :10], 42.0, atol=1e-4)    # filled, then standardised with mean 0 / std 1


def test_water_occurrence_is_clipped_to_0_100():
    img = _image()
    img[11] = 250.0
    out = preprocess_12ch_benchmark(img, STATS)
    assert np.allclose(out[11], 100.0, atol=1e-4)


def test_wrong_shape_raises():
    with pytest.raises(ValueError):
        preprocess_12ch_benchmark(np.zeros((7, 128, 128), dtype=np.float32), STATS)


def test_nan_input_raises():
    img = _image()
    img[3, 0, 0] = np.nan
    with pytest.raises(ValueError):
        preprocess_12ch_benchmark(img, STATS)


def test_rgb_composite_range():
    rgb = make_rgb_composite(np.random.rand(*IMAGE_SHAPE).astype(np.float32) * 1000)
    assert rgb.shape == (128, 128, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
