
"""WaterSeg AI — Flask REST API and Gallery Backend."""

from io import BytesIO
from pathlib import Path
import base64
import json

import numpy as np
import tifffile
from PIL import Image
from flask import Flask, request, jsonify, render_template

from inference import WaterSegmentationInference
from src.preprocessing import make_rgb_composite

BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    root_path=str(BASE_DIR),
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

# Load the model once at application startup.
predictor = WaterSegmentationInference(device="cpu")

ALLOWED_EXTENSIONS = {".tif", ".tiff", ".npy"}
GALLERY_DIR = BASE_DIR / "gallery_samples"
MANIFEST_PATH = BASE_DIR / "static" / "gallery_manifest.json"

with MANIFEST_PATH.open("r", encoding="utf-8") as file:
    GALLERY = json.load(file)

GALLERY_LOOKUP = {
    str(item["sample_id"]): item for item in GALLERY
}


def encode_png(array, mode="RGB"):
    """Encode an image as a browser-ready PNG data URL."""
    image = Image.fromarray(array.astype(np.uint8), mode=mode)
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return "data:image/png;base64," + encoded


def prepare_rgb(raw_image):
    """Generate display-only RGB from original 12-band data."""
    rgb = np.asarray(
        make_rgb_composite(raw_image),
        dtype=np.float32
    )

    if rgb.shape != (128, 128, 3):
        raise ValueError(f"Invalid RGB shape: {rgb.shape}")

    if not np.isfinite(rgb).all():
        raise ValueError("RGB preview contains invalid values")

    if rgb.min() >= 0 and rgb.max() <= 1.5:
        rgb = rgb * 255.0

    return np.clip(rgb, 0, 255).astype(np.uint8)


def make_overlay(rgb, binary_mask):
    """Highlight detected water in cyan over RGB."""
    overlay = rgb.astype(np.float32).copy()
    water = binary_mask.astype(bool)

    cyan = np.array([0, 235, 235], dtype=np.float32)
    alpha = 0.55

    overlay[water] = (
        (1 - alpha) * overlay[water]
        + alpha * cyan
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def read_uploaded_image(uploaded_file):
    extension = Path(uploaded_file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file format: use TIFF or NPY")

    file_bytes = uploaded_file.read()

    if not file_bytes:
        raise ValueError("Uploaded file is empty")

    buffer = BytesIO(file_bytes)

    if extension == ".npy":
        raw = np.load(buffer, allow_pickle=False)
    else:
        raw = tifffile.imread(buffer)

    if raw.shape == (128, 128, 12):
        raw = np.transpose(raw, (2, 0, 1))

    return raw


def prediction_response(raw_image, source, filename, sample_id=None):
    """Predict and return Original, Mask, Overlay and coverage."""
    result = predictor.predict_array(raw_image)

    mask = np.asarray(result["mask"], dtype=np.uint8)

    if mask.shape != (128, 128):
        raise ValueError(f"Invalid prediction shape: {mask.shape}")

    rgb = prepare_rgb(raw_image)
    overlay = make_overlay(rgb, mask)
    mask_png = mask * 255

    return {
        "success": True,
        "source": source,
        "filename": filename,
        "sample_id": sample_id,
        "water_percentage": round(
            float(mask.mean() * 100), 2
        ),
        "original_image": encode_png(rgb, "RGB"),
        "mask_image": encode_png(mask_png, "L"),
        "overlay_image": encode_png(overlay, "RGB"),
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "input_shape": [12, 128, 128],
        "gallery_count": len(GALLERY),
        "model": predictor.checkpoint_info
    })


@app.get("/gallery")
def gallery():
    return jsonify({
        "count": len(GALLERY),
        "samples": GALLERY
    })


@app.post("/predict-sample/<sample_id>")
def predict_sample(sample_id):
    item = GALLERY_LOOKUP.get(sample_id)

    if item is None:
        return jsonify({"error": "Sample not found"}), 404

    sample_path = GALLERY_DIR / item["filename"]

    if not sample_path.is_file():
        return jsonify({"error": "Sample file missing"}), 500

    try:
        raw = predictor.load_image(sample_path)

        return jsonify(
            prediction_response(
                raw,
                source="gallery",
                filename=item["filename"],
                sample_id=sample_id
            )
        )

    except (ValueError, TypeError, OSError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/predict")
def predict_upload():
    if "file" not in request.files:
        return jsonify({
            "error": "Missing multipart file field: file"
        }), 400

    uploaded = request.files["file"]

    if not uploaded.filename:
        return jsonify({"error": "No file selected"}), 400

    extension = Path(uploaded.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": "Unsupported file format. Use TIFF or NPY."
        }), 415

    try:
        raw = read_uploaded_image(uploaded)

        return jsonify(
            prediction_response(
                raw,
                source="upload",
                filename=uploaded.filename
            )
        )

    except (ValueError, TypeError, OSError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "File exceeds 20 MB limit"
    }), 413


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
