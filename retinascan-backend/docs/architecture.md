# Architecture

This document describes the high-level system architecture of the RetinaScan AI backend, covering the major components, their responsibilities, and how they connect.

---

## System Overview

RetinaScan AI is a **dual-model ensemble inference system** for 10-class retinal disease classification. It combines two fine-tuned convolutional neural networks — **EfficientNet-B0** and **MobileNet-V2** — by averaging their softmax probabilities at inference time. The system is composed of four distinct layers:

```
┌───────────────────────────────────────────────────────────────────┐
│                        Presentation Layer                         │
│    Streamlit Web App  ·  Swagger/ReDoc UI  ·  cURL / SDKs        │
└────────────────────────────────┬──────────────────────────────────┘
                                 │ HTTP (JSON + multipart/form-data)
┌────────────────────────────────▼──────────────────────────────────┐
│                         API Layer (FastAPI)                        │
│   /health  ·  /model/info  ·  /predict                           │
│   CORS  ·  Request logging middleware  ·  Global error handler    │
└────────────────────────────────┬──────────────────────────────────┘
                                 │
┌────────────────────────────────▼──────────────────────────────────┐
│                        Service Layer                              │
│   EnsembleInferenceService (ONNX Runtime)                         │
│   GradCAMService (PyTorch)                                        │
│   Preprocessing · Validators                                      │
└────────────────────────────────┬──────────────────────────────────┘
                                 │
┌────────────────────────────────▼──────────────────────────────────┐
│                         Model Layer                               │
│   efficientnet.onnx  ·  mobilenet.onnx  ·  ensemble_config.json  │
│   efficientnet_unfrozen_best.pth  ·  mobilenet_unfrozen_best.pth │
└───────────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. Presentation Layer

| Component | Location | Purpose |
|:---|:---|:---|
| **Streamlit App** | `retinascan-backend/streamlit_app/app.py` | Interactive browser UI for uploading fundus images, viewing predictions, model agreement badges, probability charts, and Grad-CAM heatmaps. Connects to the FastAPI backend over HTTP. |
| **Swagger / ReDoc** | Auto-generated at `/docs` and `/redoc` | OpenAPI documentation served directly by FastAPI. |

### 2. API Layer — FastAPI Application

| File | Responsibility |
|:---|:---|
| `app/main.py` | Application entrypoint. Configures the FastAPI instance, attaches the **lifespan** context manager (which loads models once at startup and tears them down on shutdown), adds CORS middleware, a request-logging HTTP middleware, and a global exception handler that prevents stack trace leaks. Mounts the two routers. |
| `app/routes/health.py` | Exposes `GET /health` (returns 200 if models are loaded, 503 otherwise) and `GET /model/info` (returns ensemble configuration, evaluation metrics, and class mappings from `ensemble_config.json`). |
| `app/routes/predict.py` | Exposes `POST /predict`. Orchestrates the full prediction pipeline: image validation → dual-ONNX ensemble inference → Grad-CAM generation → response assembly. |
| `app/schemas/prediction.py` | Pydantic models defining the response contracts (`PredictionResponse`, `HealthResponse`, `ModelInfoResponse`). Enforces type safety and auto-generates OpenAPI documentation. |
| `app/utils/validators.py` | Validates uploaded images: checks MIME type/extension, file size (≤25 MB), PIL decodability, and minimum resolution (64×64). Returns clean 4xx errors for all rejection cases. |

### 3. Service Layer

| Service | File | Runtime | Purpose |
|:---|:---|:---|:---|
| **EnsembleInferenceService** | `app/services/inference.py` | ONNX Runtime | Loads both `.onnx` models and the JSON configuration at startup. Runs dual forward passes, applies softmax, and combines probabilities using the configured strategy (simple average by default). Performs a **startup safety check** that compares each model's output dimension against the class count in `ensemble_config.json` — if they mismatch, the service refuses to start. |
| **GradCAMService** | `app/services/gradcam.py` | PyTorch | Loads the `.pth` checkpoints in eval mode. Uses forward/backward hooks on `features[-1]` to capture activations and gradients, computes the Grad-CAM heatmap, resizes it, applies a JET colormap, alpha-blends it onto the original image, and returns a base64 PNG data URI. |
| **Preprocessing** | `app/services/preprocessing.py` | NumPy + PIL | Replicates the exact torchvision training transforms: `Resize(224, 224)` → `ToTensor` (divide by 255) → `Normalize(ImageNet mean/std)`. Outputs a float32 NCHW array. |

### 4. Model Layer

| Artifact | Format | Size | Purpose |
|:---|:---|:---|:---|
| `efficientnet.onnx` | ONNX | 15.3 MB | Exported EfficientNet-B0 with modified classifier head (10 outputs). |
| `mobilenet.onnx` | ONNX | 8.5 MB | Exported MobileNet-V2 with modified classifier head (10 outputs). |
| `ensemble_config.json` | JSON | ~2 KB | Single source of truth: class mappings, combination method, weights, solo/ensemble F1 scores, and low-support class list. |
| `efficientnet_unfrozen_best.pth` | PyTorch checkpoint | 15.6 MB | Full PyTorch state dict — used only by GradCAMService (ONNX Runtime cannot compute gradients). |
| `mobilenet_unfrozen_best.pth` | PyTorch checkpoint | 8.8 MB | Full PyTorch state dict for MobileNet Grad-CAM. |

---

## Why Two Runtimes?

- **ONNX Runtime** serves production inference because it is significantly faster on CPU, supports graph-level optimizations, and has a smaller memory footprint than PyTorch.
- **PyTorch** is loaded solely for Grad-CAM because the technique requires backpropagation through intermediate convolutional layers to compute gradient-weighted class activation maps — something ONNX Runtime does not support.

This dual-runtime approach isolates the performance-critical inference path (ONNX) from the explainability path (PyTorch), so the Grad-CAM overhead does not affect core inference latency.

---

## Key Design Decisions

1. **Lifespan-based model loading**: Models are loaded once at application startup via FastAPI's `lifespan` context manager and stored on `app.state`. This avoids per-request model loading and ensures clean shutdown.

2. **Startup safety validation**: The `EnsembleInferenceService` compares ONNX output dimensions against the class count in `ensemble_config.json` at startup. If a model was re-exported with a different number of classes but the config was not updated, the service fails immediately with a descriptive error rather than silently producing misaligned predictions.

3. **Configuration-driven ensemble**: The combination strategy, class mappings, and reliability metadata are all read from `ensemble_config.json`, making it trivial to switch between simple average, weighted average, or solo-model modes without code changes.

4. **Low-support reliability flagging**: Classes with fewer than 25 test samples are listed in the configuration and automatically flagged in API responses, alerting downstream consumers that the prediction may be statistically unreliable.

5. **Graceful degradation**: If the PyTorch checkpoints are not present on disk, Grad-CAM is simply disabled (logged as a warning) and the `/predict` endpoint returns an empty overlay string instead of crashing.
