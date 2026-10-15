
"""Week 3 - Multispectral Water Segmentation Inference."""

from pathlib import Path
import sys
import numpy as np
import torch
import tifffile
import segmentation_models_pytorch as smp

PROJECT_DIR = Path(__file__).resolve().parent

# Import the original Week 2 preprocessing.
sys.path.insert(0, str(PROJECT_DIR))

from src.preprocessing import (
    preprocess_12ch_benchmark,
    make_rgb_composite,
)
from src.data import load_stats

IMAGE_SHAPE = (12, 128, 128)
THRESHOLD = 0.5


class WaterSegmentationInference:

    def __init__(
        self,
        checkpoint_path=None,
        stats_path=None,
        device=None
    ):
        self.device = torch.device(
            device or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        checkpoint_path = Path(
            checkpoint_path
            or PROJECT_DIR / "best_water_segmentation_model.pth"
        )

        stats_path = Path(
            stats_path
            or PROJECT_DIR / "config" / "stats.json"
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint missing: {checkpoint_path}"
            )

        if not stats_path.is_file():
            raise FileNotFoundError(
                f"Training statistics missing: {stats_path}"
            )

        self.stats = load_stats(stats_path)

        # Load the original trained checkpoint.
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        assert checkpoint["encoder"] == "resnet34"
        assert checkpoint["decoder"] == "unet"
        assert checkpoint["in_channels"] == 12

        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=12,
            classes=1,
            activation=None
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )

        self.model.to(self.device)
        self.model.eval()

        self.checkpoint_info = {
            "encoder": checkpoint["encoder"],
            "decoder": checkpoint["decoder"],
            "strategy": checkpoint["strategy"],
            "seed": checkpoint["seed"],
            "best_epoch": checkpoint["best_epoch"]
        }

    @staticmethod
    def load_image(image_path):
        """Read a genuine 12-band TIFF or NumPy image."""

        path = Path(image_path)

        if path.suffix.lower() in (".tif", ".tiff"):
            image = tifffile.imread(path)

        elif path.suffix.lower() == ".npy":
            image = np.load(path, allow_pickle=False)

        else:
            raise ValueError(
                "Unsupported format. Use .tif, .tiff or .npy"
            )

        # Accept channel-last or channel-first layout.
        if image.shape == (128, 128, 12):
            image = np.transpose(image, (2, 0, 1))

        if image.shape != IMAGE_SHAPE:
            raise ValueError(
                f"Expected 12-channel image {IMAGE_SHAPE}; "
                f"received {image.shape}"
            )

        image = image.astype(np.float32)

        if not np.isfinite(image).all():
            raise ValueError("Input contains NaN or Inf")

        return image

    def predict_array(self, raw_image):
        """Predict a binary water mask from a raw 12-band array."""

        raw_image = np.asarray(raw_image)

        if raw_image.shape != IMAGE_SHAPE:
            raise ValueError(
                f"Expected {IMAGE_SHAPE}, got {raw_image.shape}"
            )

        if not np.isfinite(raw_image).all():
            raise ValueError("Input contains NaN or Inf")

        # EXACT original Week 2 preprocessing.
        processed = preprocess_12ch_benchmark(
            raw_image, self.stats
        )

        tensor = torch.from_numpy(
            np.ascontiguousarray(processed)
        ).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.sigmoid(logits)

            binary_mask = (
                probabilities >= THRESHOLD
            ).squeeze().cpu().numpy().astype(np.uint8)

        if binary_mask.shape != (128, 128):
            raise RuntimeError("Invalid prediction shape")

        return {
            "mask": binary_mask,
            "water_percentage": float(
                binary_mask.mean() * 100
            ),
            "rgb_preview": make_rgb_composite(raw_image)
        }

    def predict_file(self, image_path):
        raw_image = self.load_image(image_path)
        return self.predict_array(raw_image)

    @staticmethod
    def save_mask(mask, output_path):
        """Save a binary mask as a visible black-and-white PNG."""
        from PIL import Image

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        mask = np.asarray(mask)

        if mask.shape != (128, 128):
            raise ValueError("Invalid mask shape")

        if not np.isin(mask, [0, 1]).all():
            raise ValueError("Mask must contain only 0 and 1")

        Image.fromarray(
            (mask * 255).astype(np.uint8),
            mode="L"
        ).save(output_path)

        return output_path
