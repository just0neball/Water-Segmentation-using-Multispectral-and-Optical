> **GitHub setup:** Download the trained checkpoint
> from the repository's Releases section and save it
> as `week3/best_water_segmentation_model.pth`
> before running the application.
>
> See `MODEL_SETUP.md` for details.

# WaterSeg AI
## Week 3 — Multispectral Water Segmentation Deployment

WaterSeg AI is a local web application for identifying
water regions in multispectral satellite images.

## Model

- Architecture: ResNet34 U-Net
- Framework: PyTorch
- Input: 12 x 128 x 128
- Output: Binary water segmentation mask
- Threshold: 0.5
- Device: CPU
- Initialization: rgb_zero_extra
- Random seed: 123
- Selected best epoch: 61
- Week 2 best validation IoU: approximately 0.7867

## Application Features

1. Gallery containing all 46 held-out Test Set images.
2. Search images by sample number.
3. Select and analyze a gallery image.
4. Upload an external TIFF or NPY multispectral image.
5. Display:
   - Original Satellite RGB Preview
   - Predicted Binary Water Mask
   - Water Detection Overlay
6. Calculate predicted water coverage.
7. Download Original, Mask and Overlay PNG images.

Water Coverage is NOT model accuracy.

## Windows Setup

Requirements:
- Windows computer
- Python 3.10 or newer
- Internet access for the first dependency installation

Steps:

1. Extract WaterSeg_AI_Gallery_Local.zip.
2. Open the extracted folder.
3. Double-click run.bat.
4. Wait for package installation and model loading.
5. Open http://127.0.0.1:5000 in your browser.
6. Keep the command window open while using the app.

If an older WaterSeg AI server is running on port 5000,
stop it first using Ctrl+C in its command window.

## Gallery Testing

The gallery contains 46 images from the original Test Set.

Select any image and click Analyze Selected Image.

You can also open the gallery_samples folder and upload
one of its original TIFF files through the upload section.

## External Upload

Supported formats:
- .tif
- .tiff
- .npy

Expected image shape:
- (12, 128, 128), or
- (128, 128, 12)

Maximum upload size: 20 MB.

The spectral band order and image value conventions must
match the original training dataset.

Ordinary RGB/JPEG images are not supported.

## REST API

GET /
    Web dashboard

GET /health
    Service health and model information

GET /gallery
    List all 46 available gallery samples

POST /predict
    Multipart upload using field name: file
    Returns JSON with prediction and image data URLs

POST /predict-sample/<sample_id>
    Predict a built-in gallery sample
    Returns JSON with prediction and image data URLs

## Output

- Original preview: RGB PNG
- Binary mask: grayscale PNG
- Overlay: RGB PNG highlighting predicted water in cyan

Mask interpretation:
- White = predicted water
- Black = background

## Verified Integration Tests

- Homepage, CSS, JavaScript and health endpoint
- All 46 gallery images and previews
- Three real gallery predictions: 261, 161, 245
- External TIFF upload
- External NPY upload
- Prediction matches standalone inference
- Nine API error-handling cases

The Flask test client was used for integration testing.
Interactive browser testing is performed separately.

## Example Test Results

Sample 261:
- Predicted coverage: 0.55%
- IoU: 0.1845

Sample 161:
- Predicted coverage: 39.70%
- IoU: 0.7196

Sample 245:
- Predicted coverage: 55.34%
- IoU: 0.9317

These results describe individual test samples,
not overall test-set accuracy.

## Project Notes

- Original Week 2 preprocessing is reused.
- Original training statistics are included.
- The selected trained checkpoint is included.
- No retraining is required for local inference.
- The app uses Flask's development server.
- Do not expose this local development server publicly.
- GitHub publishing is intentionally deferred.
