# RetinaScan AI — Backend Service
### Dual-Backbone ONNX Ensemble with Grad-CAM Explainability for 10-Class Retinal Disease Detection

RetinaScan AI is a production-grade inference service that classifies retinal fundus camera images into 10 ocular condition categories (including Diabetic Retinopathy, Glaucoma, and Myopia). It executes a verified dual-CNN ensemble combining EfficientNet-B0 and MobileNet-V2 via ONNX Runtime for high throughput and low latency, accompanied by visual Grad-CAM heatmaps for interpretable clinical AI.

---

## 🏛️ System Architecture

```
[ Uploaded Fundus Image ]
          │
          ▼
 [ Preprocessing Pipeline ] (Resize 224x224, ImageNet Normalization)
          │
    ┌─────┴────────────────────────┐
    │                              │
    ▼                              ▼
[ EfficientNet-B0 ONNX ]   [ MobileNet-V2 ONNX ]
    │                              │
    └─────┬────────────────────────┘
          │ (Softmax Probabilities)
          ▼
[ Ensemble Combiner ] ──▶ (Configured Winner: Simple Average)
          │
    ┌─────┴────────────────────────┐
    │                              │
    ▼                              ▼
[ Diagnostic Prediction ]    [ PyTorch Grad-CAM Hook ]
- Top Class & Confidence     - Feature Activation Overlays
- Model Agreement Check      - Base64 Heatmap Output
- Low-Support Warning
```

---

## 📊 Model Evaluation Summary

- **Winning Strategy**: **Simple-Average Ensemble** with **0.705 Macro-F1** and **71.0% Accuracy** on 810 held-out test images.
- **Ensemble Gain**: **+0.080 (+8.0% Macro-F1)** improvement over the best solo model (EfficientNet-B0 at 0.625).
- **ONNX Numerical Parity**: Verified with maximum absolute probability deviation $< 4.5 \times 10^{-5}$ against PyTorch.
- **Detailed Evaluation**: See full writeup, confusion matrix analysis, and per-class breakdown in [docs/model_evaluation.md](docs/model_evaluation.md).

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.11+
- Virtual environment tool (`venv` or `conda`)
- (Optional) Docker

### 2. Local Installation

```bash
# Navigate to the backend directory
cd retinascan-backend

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Model Artifacts Placement

Ensure the model files are located under `ml/models/`:
- `ml/models/efficientnet.onnx`
- `ml/models/mobilenet.onnx`
- `ml/models/ensemble_config.json`

*(To re-export ONNX from `.pth` checkpoints, run `python ml/export/export_effnet.py` and `python ml/export/export_mobilenet.py`)*.

### 4. Running the Server Locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The interactive API documentation will be available at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🐳 Docker Deployment

```bash
# Build the Docker image
docker build -t retinascan-backend .

# Run the container
docker run -d -p 8000:8000 --name retinascan-api retinascan-backend

# Check container health status
docker ps
```

---

## 📡 API Reference

### 1. Health Check
`GET /health`
```json
{
  "status": "healthy",
  "models_loaded": true,
  "version": "1.0.0"
}
```

### 2. Model Configuration & Evaluation Info
`GET /model/info`
```json
{
  "combination_method": "simple_average",
  "weights": { "efficientnet": 0.5, "mobilenet": 0.5 },
  "solo_test_macro_f1": { "efficientnet": 0.624959, "mobilenet": 0.615006 },
  "ensemble_test_macro_f1": 0.705040,
  "low_support_classes": ["Central Serous Chorioretinopathy", "Disc Edema", "Pterygium", "Retinal Detachment", "Retinitis Pigmentosa"],
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

### 3. Predict & Explain
`POST /predict` (Multipart Form-data with file `file`)

**Example Request (cURL)**:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@retinal_image.jpg"
```

**Example Response**:
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
    "gradcam_overlay_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA..."
  },
  "meta": {
    "inference_time_ms": 38.45,
    "model_version": "1.0.0"
  },
  "disclaimer": "Research prototype — not a certified diagnostic tool."
}
```

---

## 🧪 Running Tests

```bash
# Run pytest across API and inference test suites
pytest -v
```
