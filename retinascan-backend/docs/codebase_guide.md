# Codebase Guide

A file-by-file walkthrough of the RetinaScan AI repository, explaining what each file does and how it fits into the overall system.

---

## Root Directory

```
AI-Bases-Eye-Disease-Detection-Using-Fundus-Camera-Images/
```

| File / Directory | Description |
|:---|:---|
| `README.md` | Project overview, quickstart guide, and usage instructions. |
| `IMPLEMENTATION_PLAN (4).md` | The 8-phase build specification that guided the construction of the backend. |
| `fundus-images-training.ipynb` | Kaggle training notebook — data loading, augmentation, model training, evaluation, confusion matrices, and ensemble comparison. |
| `.gitignore` | Excludes `venv/`, `__pycache__/`, `.env`, logs, and editor files from version control. |
| `training-artifacts/` | Contains trained model checkpoints, evaluation results JSON files, and the 810-image held-out test set. |
| `retinascan-backend/` | The production backend service — all application code, models, tests, docs, and deployment config. |

---

## `training-artifacts/`

These are the outputs of the training notebook. They are consumed by the backend at build time (ONNX export) and at runtime (Grad-CAM).

| File | Description |
|:---|:---|
| `efficientnet_unfrozen_best.pth` | PyTorch checkpoint for the fine-tuned EfficientNet-B0 (15.6 MB). Contains `model_state_dict`, `num_classes`, and training metadata. |
| `mobilenet_unfrozen_best.pth` | PyTorch checkpoint for the fine-tuned MobileNet-V2 (8.8 MB). Same format. |
| `eval_baseline_results.json` | Per-model test metrics (accuracy, F1, precision, recall) and the class-to-index mapping used during training. |
| `ensemble_results.json` | Ensemble evaluation results — simple average vs weighted average macro-F1 on the test set. |
| `test-images/` | 810 test images organized into 10 subdirectories, one per disease class. Used for ONNX parity verification and Grad-CAM sample generation. |

---

## `retinascan-backend/`

The production service. Everything below is relative to this directory.

### Top-Level Files

| File | Description |
|:---|:---|
| `requirements.txt` | Python dependency list: `torch`, `torchvision`, `onnxruntime`, `fastapi`, `uvicorn`, `pydantic`, `pillow`, `opencv-python-headless`, `numpy`, `matplotlib`, `pytest`, `httpx`. |
| `.gitignore` | Backend-specific ignores: `.onnx` model binaries, `.pth` checkpoints, virtual environments. |
| `Dockerfile` | Production container image based on `python:3.11-slim`. Installs system dependencies for OpenCV, copies `app/` and `ml/models/`, exposes port 8000, and runs Uvicorn. Includes a Docker `HEALTHCHECK` against `/health`. |
| `.dockerignore` | Excludes `tests/`, `docs/`, `ml/export/`, `.pth` files, and `venv/` from the Docker build context. |
| `README.md` | Backend-specific documentation with architecture diagram, evaluation summary, API reference, and deployment instructions. |

---

### `app/` — FastAPI Application

#### `app/main.py`
The entrypoint. Creates the `FastAPI` instance with a **lifespan** context manager that loads both the `EnsembleInferenceService` (ONNX) and `GradCAMService` (PyTorch) into `app.state` at startup. Configures:
- **CORS middleware** — allows all origins for frontend access.
- **HTTP request logging middleware** — logs method, path, status code, and latency for every request.
- **Global exception handler** — catches unhandled exceptions and returns a clean 500 JSON response instead of leaking stack traces.

Mounts two routers: `health_router` and `predict_router`.

#### `app/routes/health.py`
Two read-only endpoints:
- `GET /health` — returns `{"status": "healthy", "models_loaded": true}` (200) or 503 if the inference service failed to load.
- `GET /model/info` — returns the full ensemble configuration including combination method, solo/ensemble F1 scores, class mappings, and the list of low-support classes.

#### `app/routes/predict.py`
The core inference endpoint:
- `POST /predict` — accepts a multipart file upload, validates it, runs dual-ONNX ensemble inference, generates a Grad-CAM overlay via PyTorch, and returns a structured `PredictionResponse` containing the disease prediction, confidence, per-class probabilities, model agreement check, reliability flag, Grad-CAM base64 image, latency, and a medical disclaimer.

#### `app/schemas/prediction.py`
Pydantic `BaseModel` classes that define the API response contracts:
- `PredictionDetail` — disease name, confidence, combination method, and full probability distribution.
- `ModelAgreement` — solo predictions from each backbone and whether they agree.
- `ReliabilityFlag` — boolean flag and optional advisory note for low-support classes.
- `Explainability` — base64-encoded Grad-CAM overlay PNG.
- `PredictionMeta` — inference latency and model version.
- `PredictionResponse` — top-level response composing all of the above plus a disclaimer.
- `HealthResponse` and `ModelInfoResponse` — schemas for the health and model info endpoints.

#### `app/services/preprocessing.py`
A single function `preprocess_fundus_image(image_input)` that replicates the exact torchvision transforms used during training:
1. Open image from bytes or PIL object, convert to RGB.
2. Resize to 224×224 with bilinear interpolation.
3. Convert to float32, divide by 255 (equivalent to `ToTensor()`).
4. Transpose to NCHW layout.
5. Normalize with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.

Returns a `(1, 3, 224, 224)` float32 NumPy array ready for ONNX Runtime.

#### `app/services/inference.py`
`EnsembleInferenceService` — the core inference engine:
- **Constructor**: loads `ensemble_config.json`, creates two `onnxruntime.InferenceSession` objects (CPU provider, full graph optimization), and runs a startup safety validation that checks each model's output dimension against the config's class count.
- **`predict(raw_image_input)`**: preprocesses the image, runs both ONNX sessions, applies softmax, combines probabilities (simple average / weighted average / solo), identifies the top class, checks model agreement, flags low-support classes, and returns a structured dictionary.

#### `app/services/gradcam.py`
Two classes:
- **`GradCAM`** — a lightweight, self-contained Grad-CAM implementation. Registers forward and backward hooks on a target convolutional layer to capture activations and gradients. Computes the heatmap as: global-average-pool the gradients → multiply by activations → sum over channels → ReLU → normalize to [0, 1].
- **`GradCAMService`** — loads PyTorch `.pth` checkpoints, reconstructs the model architectures with modified classifier heads, and exposes `generate_gradcam_overlay()` which runs the Grad-CAM computation, resizes the heatmap, applies a JET colormap, alpha-blends it onto the original image, and returns a base64 PNG data URI.

#### `app/utils/validators.py`
`validate_image_file(file)` — an async function that validates uploaded images through four checks:
1. MIME type or file extension against an allowlist.
2. File size ≤ 25 MB.
3. PIL `verify()` for structural integrity (catches truncated/corrupted files).
4. Minimum resolution of 64×64 pixels.

Returns `(image_bytes, PIL.Image)` on success; raises `HTTPException` with a descriptive 4xx error on failure.

---

### `ml/` — Model Export & Verification

#### `ml/models/`
Contains the production model artifacts:
- `efficientnet.onnx` — exported ONNX model (15.3 MB).
- `mobilenet.onnx` — exported ONNX model (8.5 MB).
- `ensemble_config.json` — the single source of truth for class mappings, combination method, weights, test metrics, and low-support class list.

#### `ml/export/export_effnet.py`
Script that loads `efficientnet_unfrozen_best.pth`, reconstructs EfficientNet-B0 with a 10-class classifier head, and exports to ONNX using `torch.onnx.export` with `dynamo=False` (to avoid a Windows charmap encoding bug).

#### `ml/export/export_mobilenet.py`
Same as above, but for MobileNet-V2.

#### `ml/export/verify_onnx_export.py`
Runs both the PyTorch and ONNX models on all 810 real test images and verifies:
- Maximum absolute probability difference is below `1e-4` (numerical parity).
- Test Macro-F1 scores match exactly between PyTorch and ONNX.

#### `ml/export/generate_ensemble_config.py`
Reads the training artifacts JSON files, evaluates all four combination strategies (EfficientNet solo, MobileNet solo, weighted average, simple average), picks the winner, and writes `ensemble_config.json`.

#### `ml/export/verify_gradcam.py`
Generates 12 sample Grad-CAM heatmap images (3 Glaucoma + 3 Diabetic Retinopathy × 2 backbones) and saves them to `docs/gradcam_samples/` for visual inspection.

---

### `tests/`

| File | Tests | Description |
|:---|:---|:---|
| `test_inference.py` | 5 tests | Startup safety validation with mismatched class count, preprocessing output shape/range, real image prediction structure, PIL Image input, and solo override behavior. |
| `test_api.py` | 6 tests | Health endpoint 200 check, model info metadata validation, full `/predict` response structure with a real image, rejection of non-image files (415), rejection of corrupted JPEG (400), and rejection of sub-minimum-resolution images (400). |

All 11 tests use FastAPI's `TestClient` which exercises the real lifespan loading and middleware stack.

---

### `docs/`

| File | Description |
|:---|:---|
| `model_evaluation.md` | Detailed empirical evaluation: test setup, per-class F1/precision/recall table, ensemble gain analysis, shared failure modes (Glaucoma confusion), low-support flagging rationale, and ONNX parity verification results. |
| `architecture.md` | System architecture documentation (this companion document). |
| `codebase_guide.md` | This file — file-by-file codebase walkthrough. |
| `data_flow.md` | End-to-end data flow from image upload to JSON response. |
| `api.md` | Complete API reference with request/response schemas and examples. |
| `gradcam_samples/` | 12 generated Grad-CAM overlay PNGs for visual verification. |

---

### `streamlit_app/`

| File | Description |
|:---|:---|
| `app.py` | Interactive Streamlit web application. Connects to the FastAPI backend, shows live health status, model evaluation benchmarks, and low-support class warnings in the sidebar. Main area supports image upload and sample test image selection, displays prediction results with a confidence progress bar, dual-model agreement badges, reliability warnings, Grad-CAM heatmap rendering, and a bar chart of all 10 class probabilities. |
