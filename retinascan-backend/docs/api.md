# API Reference

Complete reference for all HTTP endpoints exposed by the RetinaScan AI backend.

**Base URL**: `http://localhost:8000`
**Interactive Documentation**: `http://localhost:8000/docs` (Swagger) · `http://localhost:8000/redoc` (ReDoc)

---

## Endpoints Overview

| Method | Path | Summary |
|:---|:---|:---|
| `GET` | `/health` | Service health check |
| `GET` | `/model/info` | Model configuration and evaluation metadata |
| `POST` | `/predict` | Predict retinal disease with Grad-CAM explainability |

---

## `GET /health`

Returns the current health status of the backend service.

### Response (200 OK — Healthy)

```json
{
  "status": "healthy",
  "models_loaded": true,
  "version": "1.0.0"
}
```

### Response (503 Service Unavailable — Unhealthy)

Returned when the inference service failed to initialize at startup.

```json
{
  "status": "unhealthy",
  "models_loaded": false,
  "version": "1.0.0"
}
```

### Response Schema

| Field | Type | Description |
|:---|:---|:---|
| `status` | `string` | `"healthy"` or `"unhealthy"` |
| `models_loaded` | `boolean` | Whether ONNX models were successfully loaded |
| `version` | `string` | API version |

---

## `GET /model/info`

Returns model configuration, evaluation benchmarks, and class mappings from `ensemble_config.json`.

### Response (200 OK)

```json
{
  "combination_method": "simple_average",
  "weights": {
    "efficientnet": 0.5,
    "mobilenet": 0.5
  },
  "solo_test_macro_f1": {
    "efficientnet": 0.624959,
    "mobilenet": 0.615006
  },
  "ensemble_test_macro_f1": 0.705040,
  "low_support_classes": [
    "Central Serous Chorioretinopathy",
    "Disc Edema",
    "Pterygium",
    "Retinal Detachment",
    "Retinitis Pigmentosa"
  ],
  "class_to_idx": {
    "Central Serous Chorioretinopathy": 0,
    "Diabetic Retinopathy": 1,
    "Disc Edema": 2,
    "Glaucoma": 3,
    "Healthy": 4,
    "Macular Scar": 5,
    "Myopia": 6,
    "Pterygium": 7,
    "Retinal Detachment": 8,
    "Retinitis Pigmentosa": 9
  }
}
```

### Response Schema

| Field | Type | Description |
|:---|:---|:---|
| `combination_method` | `string` | Active ensemble strategy (`"simple_average"`, `"weighted_average"`, `"efficientnet_solo"`, or `"mobilenet_solo"`) |
| `weights` | `object \| null` | Model weights if applicable (e.g. `{"efficientnet": 0.5, "mobilenet": 0.5}`) |
| `solo_test_macro_f1` | `object` | Per-backbone macro-F1 scores from test evaluation |
| `ensemble_test_macro_f1` | `float \| null` | Ensemble macro-F1 on the 810-image test set |
| `low_support_classes` | `string[]` | Classes with fewer than 25 test samples |
| `class_to_idx` | `object` | Mapping from disease name to class index |

---

## `POST /predict`

Upload a retinal fundus image for dual-model ensemble classification and Grad-CAM visual explainability.

### Request

**Content-Type**: `multipart/form-data`

| Parameter | Type | Required | Description |
|:---|:---|:---:|:---|
| `file` | `UploadFile` | ✅ | Fundus camera image in JPEG, PNG, BMP, or WEBP format |

**Constraints**:
- Maximum file size: 25 MB
- Minimum resolution: 64×64 pixels
- File must be a valid, non-corrupted image decodable by PIL

### Example Request (cURL)

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@retinal_fundus.jpg"
```

### Example Request (Python)

```python
import httpx

with open("retinal_fundus.jpg", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/predict",
        files={"file": ("fundus.jpg", f, "image/jpeg")},
    )
print(response.json())
```

### Response (200 OK)

```json
{
  "prediction": {
    "disease": "Diabetic Retinopathy",
    "confidence": 0.9412,
    "combination_method": "simple_average",
    "class_probabilities": {
      "Central Serous Chorioretinopathy": 0.0012,
      "Diabetic Retinopathy": 0.9412,
      "Disc Edema": 0.0031,
      "Glaucoma": 0.0124,
      "Healthy": 0.0152,
      "Macular Scar": 0.0089,
      "Myopia": 0.0102,
      "Pterygium": 0.0002,
      "Retinal Detachment": 0.0041,
      "Retinitis Pigmentosa": 0.0035
    }
  },
  "model_agreement": {
    "efficientnet_prediction": "Diabetic Retinopathy",
    "mobilenet_prediction": "Diabetic Retinopathy",
    "agree": true
  },
  "reliability_flag": {
    "is_low_support_class": false,
    "note": null
  },
  "explainability": {
    "gradcam_overlay_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg..."
  },
  "meta": {
    "inference_time_ms": 38.45,
    "model_version": "1.0.0"
  },
  "disclaimer": "Research prototype — not a certified diagnostic tool."
}
```

### Response Schema

#### Top-Level `PredictionResponse`

| Field | Type | Description |
|:---|:---|:---|
| `prediction` | `PredictionDetail` | Classification result and probability distribution |
| `model_agreement` | `ModelAgreement` | Solo predictions from each backbone and consensus check |
| `reliability_flag` | `ReliabilityFlag` | Statistical reliability advisory |
| `explainability` | `Explainability` | Grad-CAM visualization |
| `meta` | `PredictionMeta` | Latency and versioning metadata |
| `disclaimer` | `string` | Medical/regulatory disclaimer |

#### `PredictionDetail`

| Field | Type | Constraints | Description |
|:---|:---|:---|:---|
| `disease` | `string` | — | Name of the predicted retinal condition |
| `confidence` | `float` | `[0.0, 1.0]` | Ensemble confidence for the predicted class |
| `combination_method` | `string` | — | Strategy used (e.g. `"simple_average"`) |
| `class_probabilities` | `object` | 10 entries, sum ≈ 1.0 | Full probability distribution across all 10 classes |

#### `ModelAgreement`

| Field | Type | Description |
|:---|:---|:---|
| `efficientnet_prediction` | `string` | EfficientNet-B0's solo top-1 prediction |
| `mobilenet_prediction` | `string` | MobileNet-V2's solo top-1 prediction |
| `agree` | `boolean` | `true` if both backbones predict the same class |

#### `ReliabilityFlag`

| Field | Type | Description |
|:---|:---|:---|
| `is_low_support_class` | `boolean` | `true` if the predicted class has <25 test samples |
| `note` | `string \| null` | Advisory message when `is_low_support_class` is `true` |

#### `Explainability`

| Field | Type | Description |
|:---|:---|:---|
| `gradcam_overlay_base64` | `string` | Base64-encoded PNG data URI of the Grad-CAM heatmap overlaid on the original image. Prefix: `data:image/png;base64,` |

#### `PredictionMeta`

| Field | Type | Description |
|:---|:---|:---|
| `inference_time_ms` | `float` | Total end-to-end latency including validation, inference, and Grad-CAM (milliseconds) |
| `model_version` | `string` | Deployed model artifacts version |

---

## Error Responses

All error responses follow this format:

```json
{
  "detail": "Human-readable error message describing what went wrong."
}
```

### Client Errors (4xx)

| Status Code | Condition | Example `detail` |
|:---:|:---|:---|
| **400** | Empty upload | `"Uploaded file is empty (0 bytes)."` |
| **400** | Corrupted image | `"Uploaded file is corrupted or not a valid decodable image: ..."` |
| **400** | Resolution too low | `"Image resolution (16x16) is below the required minimum resolution of 64x64 pixels for retinal fundus analysis."` |
| **413** | File too large | `"File exceeds maximum allowed size of 25MB."` |
| **415** | Wrong file type | `"Unsupported file type 'text/plain'. Allowed types: JPEG, PNG, BMP, WEBP."` |

### Server Errors (5xx)

| Status Code | Condition | Response Body |
|:---:|:---|:---|
| **500** | Unhandled exception | `{"error": "Internal server error", "message": "An unexpected error occurred during processing.", "path": "/predict"}` |
| **503** | Models not loaded | `{"detail": "Inference service is not initialized."}` |

---

## Authentication & Rate Limiting

The current version does not implement authentication or rate limiting. CORS is configured to allow all origins (`*`). For production deployment, consider:
- Adding API key authentication via a middleware or dependency.
- Implementing rate limiting with a library like `slowapi`.
- Restricting CORS origins to known frontend domains.
