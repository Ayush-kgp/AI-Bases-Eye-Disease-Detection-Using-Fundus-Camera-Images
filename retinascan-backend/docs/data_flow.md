# Data Flow

This document traces the complete journey of data through the RetinaScan AI system — from the moment a fundus image enters the system to the final JSON response returned to the client.

---

## End-to-End Flow Diagram

```
Client (Streamlit / cURL / SDK)
   │
   │  POST /predict  [multipart/form-data: file=fundus.jpg]
   ▼
┌──────────────────────────────────────────────────────────────────┐
│  FastAPI Middleware Stack                                         │
│  1. CORS Middleware (adds Access-Control headers)                 │
│  2. Request Logging Middleware (starts timer)                     │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  Route Handler: POST /predict  (app/routes/predict.py)           │
│                                                                  │
│  Step 1: Retrieve services from app.state                        │
│          ├── inference_service (EnsembleInferenceService)         │
│          └── gradcam_service   (GradCAMService)                  │
│                                                                  │
│  Step 2: Validate uploaded image ──────────────────────┐         │
│                                                        ▼         │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  validate_image_file()  (app/utils/validators.py)        │    │
│  │  ① Check MIME type / extension against allowlist          │    │
│  │  ② Read bytes, check for empty file                       │    │
│  │  ③ Enforce 25 MB max file size                            │    │
│  │  ④ PIL.Image.verify() — structural integrity check        │    │
│  │  ⑤ Re-open image, check ≥ 64×64 resolution               │    │
│  │  → Returns (image_bytes, PIL.Image) or raises HTTPException│   │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Step 3: Run ensemble inference ───────────────────────┐         │
│                                                        ▼         │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  inference_service.predict(image_bytes)                   │    │
│  │                                                           │    │
│  │  A. Preprocess (app/services/preprocessing.py)            │    │
│  │     ① Open bytes as PIL RGB image                         │    │
│  │     ② Resize to 224×224 (bilinear)                        │    │
│  │     ③ Convert to float32, divide by 255                   │    │
│  │     ④ Transpose to NCHW (1, 3, 224, 224)                  │    │
│  │     ⑤ Normalize with ImageNet mean/std                    │    │
│  │     → Returns float32 NumPy array                         │    │
│  │                                                           │    │
│  │  B. EfficientNet-B0 ONNX forward pass                     │    │
│  │     session.run(input_tensor) → logits (1, 10)            │    │
│  │     Apply softmax → effnet_probs (10,)                    │    │
│  │                                                           │    │
│  │  C. MobileNet-V2 ONNX forward pass                        │    │
│  │     session.run(input_tensor) → logits (1, 10)            │    │
│  │     Apply softmax → mobilenet_probs (10,)                 │    │
│  │                                                           │    │
│  │  D. Combine probabilities (configured strategy)            │    │
│  │     simple_average: (eff + mob) / 2                       │    │
│  │     weighted_average: w_eff × eff + w_mob × mob           │    │
│  │     → combined_probs (10,), normalized to sum=1           │    │
│  │                                                           │    │
│  │  E. Extract results                                        │    │
│  │     ├── predicted_class = argmax(combined_probs)           │    │
│  │     ├── confidence = max(combined_probs)                   │    │
│  │     ├── solo predictions from each backbone                │    │
│  │     ├── model agreement (agree = eff_idx == mob_idx)       │    │
│  │     └── low-support flag from ensemble_config.json         │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Step 4: Generate Grad-CAM heatmap ───────────────────┐         │
│                                                        ▼         │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  gradcam_service.generate_gradcam_overlay(image_bytes)    │    │
│  │                                                           │    │
│  │  A. Preprocess image → PyTorch tensor                     │    │
│  │  B. Forward pass through EfficientNet-B0 (PyTorch)        │    │
│  │     → Hook captures activations at features[-1]           │    │
│  │  C. Backward pass from target class score                 │    │
│  │     → Hook captures gradients at features[-1]             │    │
│  │  D. Compute Grad-CAM heatmap                              │    │
│  │     ① Global avg pool gradients → channel weights α_k     │    │
│  │     ② Weighted sum: Σ α_k × A_k → raw CAM                │    │
│  │     ③ ReLU → clip negatives                               │    │
│  │     ④ Normalize to [0, 1]                                 │    │
│  │  E. Resize heatmap to 224×224                              │    │
│  │  F. Apply JET colormap (OpenCV)                            │    │
│  │  G. Alpha-blend: 0.4 × heatmap + 0.6 × original          │    │
│  │  H. Encode overlay as base64 PNG data URI                  │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Step 5: Assemble PredictionResponse                             │
│  Step 6: Return JSON response                                    │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  Request Logging Middleware (logs status code + duration)         │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
                          HTTP Response
```

---

## Startup Data Flow

Before the first request is handled, the application goes through a one-time initialization sequence:

```
Uvicorn starts → FastAPI lifespan() context manager

1. Load ensemble_config.json
   → class_to_idx, class_to_idx_reverse, combination_method,
     weights, low_support_classes, solo_test_macro_f1

2. Create EfficientNet-B0 ONNX session
   → Load efficientnet.onnx into ONNX Runtime
   → Full graph optimization enabled

3. Create MobileNet-V2 ONNX session
   → Load mobilenet.onnx into ONNX Runtime

4. Startup safety validation
   → Check: effnet output dim == config num_classes (10)
   → Check: mobilenet output dim == config num_classes (10)
   → If mismatch → RuntimeError, service refuses to start

5. Load EfficientNet-B0 PyTorch checkpoint (.pth)
   → Reconstruct model architecture
   → Load state_dict
   → Set eval() mode

6. Load MobileNet-V2 PyTorch checkpoint (.pth)
   → Same as above

7. Store both services on app.state
   → app.state.inference_service
   → app.state.gradcam_service

8. Application is ready to accept requests
```

---

## Data Transformation Summary

The input image undergoes the following transformations through the pipeline:

| Stage | Format | Shape | Value Range |
|:---|:---|:---|:---|
| **Raw upload** | JPEG/PNG bytes | Variable | 0–255 (uint8) |
| **PIL open** | PIL RGB Image | (H, W, 3) | 0–255 |
| **Resize** | PIL RGB Image | (224, 224, 3) | 0–255 |
| **To float array** | NumPy float32 | (224, 224, 3) | 0.0–1.0 |
| **Transpose to NCHW** | NumPy float32 | (1, 3, 224, 224) | 0.0–1.0 |
| **ImageNet normalize** | NumPy float32 | (1, 3, 224, 224) | ~(−2.1 to +2.6) |
| **ONNX inference** | NumPy float32 | (1, 10) logits | Unbounded |
| **Softmax** | NumPy float32 | (10,) probabilities | 0.0–1.0 (sum=1) |
| **Ensemble combine** | NumPy float32 | (10,) probabilities | 0.0–1.0 (sum=1) |

---

## Error Flow

All error paths return clean JSON responses — the system never leaks Python tracebacks to clients.

| Error Condition | HTTP Code | Source |
|:---|:---:|:---|
| Unsupported file type | 415 | `validators.py` |
| Empty upload | 400 | `validators.py` |
| File exceeds 25 MB | 413 | `validators.py` |
| Corrupted / undecodable image | 400 | `validators.py` |
| Resolution below 64×64 | 400 | `validators.py` |
| Inference service not loaded | 503 | `predict.py` |
| Grad-CAM failure | — (graceful) | `predict.py` (falls back to empty overlay) |
| Unhandled exception | 500 | `main.py` global handler |
