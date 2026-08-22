"""FastAPI integration and endpoint tests for RetinaScan AI backend."""

import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app

BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = BASE_DIR.parent
TEST_IMAGES_DIR = WORKSPACE_ROOT / "training-artifacts" / "test-images"


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_valid_image_path():
    """Finds a valid test image."""
    for root, dirs, files in os.walk(TEST_IMAGES_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                return Path(root) / f
    raise FileNotFoundError("No test image found in training-artifacts/test-images")


def test_health_endpoint_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True
    assert "version" in data


def test_model_info_returns_expected_metadata(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "combination_method" in data
    assert data["combination_method"] == "simple_average"
    assert "solo_test_macro_f1" in data
    assert "efficientnet" in data["solo_test_macro_f1"]
    assert "mobilenet" in data["solo_test_macro_f1"]
    assert "ensemble_test_macro_f1" in data
    assert "low_support_classes" in data
    assert "class_to_idx" in data
    assert len(data["class_to_idx"]) == 10


def test_predict_with_valid_image_returns_full_response(client, sample_valid_image_path):
    with open(sample_valid_image_path, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("test_fundus.jpg", f, "image/jpeg")},
        )
    assert response.status_code == 200
    data = response.json()

    # 1. Prediction section
    assert "prediction" in data
    assert "disease" in data["prediction"]
    assert "confidence" in data["prediction"]
    assert 0.0 <= data["prediction"]["confidence"] <= 1.0
    assert "combination_method" in data["prediction"]
    assert "class_probabilities" in data["prediction"]
    assert len(data["prediction"]["class_probabilities"]) == 10

    # 2. Model agreement section
    assert "model_agreement" in data
    assert "efficientnet_prediction" in data["model_agreement"]
    assert "mobilenet_prediction" in data["model_agreement"]
    assert "agree" in data["model_agreement"]

    # 3. Reliability flag
    assert "reliability_flag" in data
    assert "is_low_support_class" in data["reliability_flag"]
    assert isinstance(data["reliability_flag"]["is_low_support_class"], bool)

    # 4. Explainability section
    assert "explainability" in data
    assert "gradcam_overlay_base64" in data["explainability"]
    assert data["explainability"]["gradcam_overlay_base64"].startswith("data:image/png;base64,")

    # 5. Meta & Disclaimer
    assert "meta" in data
    assert "inference_time_ms" in data["meta"]
    assert data["meta"]["inference_time_ms"] > 0
    assert "disclaimer" in data


def test_predict_rejects_non_image_file(client):
    fake_text_bytes = b"This is plain text and not a fundus image."
    response = client.post(
        "/predict",
        files={"file": ("document.txt", fake_text_bytes, "text/plain")},
    )
    # Should return clean 4xx (415 or 400), not a 500 crash
    assert 400 <= response.status_code < 500
    assert "detail" in response.json()


def test_predict_rejects_corrupted_image_file(client):
    corrupt_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 20  # Incomplete corrupted JPEG header
    response = client.post(
        "/predict",
        files={"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")},
    )
    # Should return clean 400 Bad Request, not a 500 crash
    assert response.status_code == 400
    assert "corrupted" in response.json()["detail"].lower()


def test_predict_rejects_tiny_low_resolution_image(client):
    from io import BytesIO
    from PIL import Image

    # Tiny 16x16 dummy image
    tiny_img = Image.new("RGB", (16, 16), color="red")
    buf = BytesIO()
    tiny_img.save(buf, format="JPEG")
    tiny_bytes = buf.getvalue()

    response = client.post(
        "/predict",
        files={"file": ("tiny.jpg", tiny_bytes, "image/jpeg")},
    )
    # Should reject below minimum resolution
    assert response.status_code == 400
    assert "minimum resolution" in response.json()["detail"].lower()
