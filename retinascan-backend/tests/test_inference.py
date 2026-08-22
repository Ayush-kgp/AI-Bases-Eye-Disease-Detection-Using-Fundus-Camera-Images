"""Unit and integration tests for EnsembleInferenceService."""

import json
import os
import tempfile
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from app.services.inference import EnsembleInferenceService
from app.services.preprocessing import preprocess_fundus_image

BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = BASE_DIR.parent
CONFIG_PATH = BASE_DIR / "ml" / "models" / "ensemble_config.json"
EFFNET_ONNX_PATH = BASE_DIR / "ml" / "models" / "efficientnet.onnx"
MOBILENET_ONNX_PATH = BASE_DIR / "ml" / "models" / "mobilenet.onnx"
TEST_IMAGES_DIR = WORKSPACE_ROOT / "training-artifacts" / "test-images"


@pytest.fixture
def real_sample_image_path():
    """Finds a real test image from test-images folder."""
    for root, dirs, files in os.walk(TEST_IMAGES_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                return Path(root) / f
    raise FileNotFoundError("No real test image found in training-artifacts/test-images")


@pytest.fixture
def real_sample_image_bytes(real_sample_image_path):
    with open(real_sample_image_path, "rb") as f:
        return f.read()


@pytest.fixture
def inference_service():
    return EnsembleInferenceService(
        config_path=CONFIG_PATH,
        effnet_onnx_path=EFFNET_ONNX_PATH,
        mobilenet_onnx_path=MOBILENET_ONNX_PATH,
    )


def test_startup_validation_fails_on_mismatched_class_count():
    """Confirms service raises a loud exception on broken class mapping."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        valid_config = json.load(f)

    # Intentionally break class count to 5 instead of 10
    broken_config = dict(valid_config)
    broken_config["class_to_idx"] = {f"Class_{i}": i for i in range(5)}
    broken_config["class_to_idx_reverse"] = {str(i): f"Class_{i}" for i in range(5)}

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(broken_config, tmp)
        tmp_path = tmp.name

    try:
        with pytest.raises(RuntimeError) as exc_info:
            EnsembleInferenceService(
                config_path=tmp_path,
                effnet_onnx_path=EFFNET_ONNX_PATH,
                mobilenet_onnx_path=MOBILENET_ONNX_PATH,
            )
        assert "STARTUP SAFETY ERROR" in str(exc_info.value)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_preprocessing_produces_correct_shape_and_range(real_sample_image_bytes):
    tensor = preprocess_fundus_image(real_sample_image_bytes)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32
    assert not np.isnan(tensor).any()
    assert not np.isinf(tensor).any()


def test_predict_real_image_returns_valid_structure(inference_service, real_sample_image_bytes):
    result = inference_service.predict(real_sample_image_bytes)

    # Verify all expected keys
    assert "predicted_class" in result
    assert "confidence" in result
    assert "combination_method" in result
    assert "all_class_probabilities" in result
    assert "solo_predictions" in result
    assert "is_low_support_class" in result
    assert "inference_time_ms" in result

    # Check types and ranges
    assert isinstance(result["predicted_class"], str)
    assert len(result["predicted_class"]) > 0
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["is_low_support_class"], bool)
    assert result["inference_time_ms"] > 0.0

    # Probabilities sum ~ 1.0
    probs = result["all_class_probabilities"]
    assert len(probs) == 10
    prob_sum = sum(probs.values())
    assert abs(prob_sum - 1.0) < 1e-4

    # Solo predictions
    solo = result["solo_predictions"]
    assert "efficientnet_prediction" in solo
    assert "mobilenet_prediction" in solo
    assert isinstance(solo["agree"], bool)


def test_predict_with_pil_image_input(inference_service, real_sample_image_path):
    pil_img = Image.open(real_sample_image_path)
    result = inference_service.predict(pil_img)
    assert result["predicted_class"] in inference_service.class_to_idx
    assert 0.0 <= result["confidence"] <= 1.0


def test_solo_override_behavior(real_sample_image_bytes):
    """Test that if config is set to solo mode, it honors that solo model."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        valid_config = json.load(f)

    solo_config = dict(valid_config)
    solo_config["combination_method"] = "efficientnet_solo"

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(solo_config, tmp)
        tmp_path = tmp.name

    try:
        service = EnsembleInferenceService(
            config_path=tmp_path,
            effnet_onnx_path=EFFNET_ONNX_PATH,
            mobilenet_onnx_path=MOBILENET_ONNX_PATH,
        )
        res = service.predict(real_sample_image_bytes)
        assert res["combination_method"] == "efficientnet_solo"
        assert res["predicted_class"] == res["solo_predictions"]["efficientnet_prediction"]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
