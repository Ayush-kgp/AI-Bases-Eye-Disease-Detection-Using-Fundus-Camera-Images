# RetinaScan AI — Fundus Eye Disease Detection System

A production-grade deep learning system for automated ocular disease detection from retinal fundus camera photographs. Powered by a dual-CNN ensemble (**EfficientNet-B0** + **MobileNet-V2**) exported to **ONNX Runtime** for high-throughput, low-latency inference with **Grad-CAM visual explainability**.

---

## 🌟 Key Capabilities & Benchmarks

- **10 Diagnostic Categories**: Diabetic Retinopathy, Glaucoma, Healthy, Myopia, Macular Scar, Retinitis Pigmentosa, Disc Edema, Retinal Detachment, Central Serous Chorioretinopathy (CSCR), and Pterygium.
- **Empirical Superiority**: The Simple-Average Ensemble achieves **0.705 Test Macro-F1** and **71.0% Accuracy** on 810 held-out test images, outperforming the best solo model by **+0.080 (+8.0% Macro-F1)**.
- **ONNX Optimization**: Numerically verified parity ($< 4.5 \times 10^{-5}$ max difference) with sub-40ms CPU inference.
- **Dual-Model Consensus & Reliability**: Real-time backbone agreement checks and automatic reliability warnings for low-support categories ($N < 25$).
- **Visual Explainability**: High-resolution Grad-CAM heatmaps overlaid directly on the original fundus image.

---

## 📁 Repository Structure

```
AI-Bases-Eye-Disease-Detection-Using-Fundus-Camera-Images/
├── retinascan-backend/             # Production FastAPI & ONNX inference service
│   ├── app/
│   │   ├── main.py                # FastAPI app with lifespan loading & CORS
│   │   ├── routes/                # Endpoints: /health, /model/info, /predict
│   │   ├── services/              # Ensemble inference, preprocessing & Grad-CAM
│   │   ├── schemas/               # Pydantic request/response models
│   │   └── utils/                 # Image validation (resolution, corruption)
│   ├── ml/
│   │   ├── export/                # ONNX export and verification scripts
│   │   └── models/                # ONNX binaries & ensemble_config.json
│   ├── docs/
│   │   ├── model_evaluation.md    # Detailed empirical evaluation report
│   │   └── gradcam_samples/       # Generated diagnostic heatmap samples
│   ├── tests/                     # 11 unit & integration tests (100% pass)
│   ├── streamlit_app/             # Modern interactive UI
│   ├── Dockerfile                 # Production Docker container
│   └── requirements.txt
├── training-artifacts/             # Trained PyTorch checkpoints & 810 test images
├── fundus-images-training.ipynb    # Kaggle training & evaluation notebook
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Ayush-kgp/AI-Bases-Eye-Disease-Detection-Using-Fundus-Camera-Images.git
cd AI-Bases-Eye-Disease-Detection-Using-Fundus-Camera-Images

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Linux/macOS

# Install backend dependencies
pip install -r retinascan-backend/requirements.txt
```

### 2. Start the Backend Service
```bash
cd retinascan-backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Launch the Streamlit Web Application
In a separate terminal:
```bash
# From the repository root
streamlit run retinascan-backend/streamlit_app/app.py
```
Open your browser at [http://localhost:8501](http://localhost:8501).

---

## 🧪 Testing

```bash
cd retinascan-backend
pytest -v
```
All 11 tests across API endpoints, model safety validation, image corruption rejection, and ONNX inference are automated.

---

## 🐳 Docker Deployment

```bash
cd retinascan-backend
docker build -t retinascan-backend .
docker run -d -p 8000:8000 --name retinascan-api retinascan-backend
```

---

## ⚕️ Medical Disclaimer

> **Research prototype for educational and technical demonstration only. Not certified as a clinical diagnostic tool or medical device. Always consult licensed medical professionals for ophthalmic diagnoses.**
