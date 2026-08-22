"""Live dual-model ensemble inference service using ONNX Runtime."""

import json
import time
from pathlib import Path
from typing import Any, Dict, Union
import numpy as np
import onnxruntime as ort
from PIL import Image

from app.services.preprocessing import preprocess_fundus_image


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


class EnsembleInferenceService:
    def __init__(
        self,
        config_path: Union[str, Path],
        effnet_onnx_path: Union[str, Path],
        mobilenet_onnx_path: Union[str, Path],
    ):
        self.config_path = Path(config_path)
        self.effnet_onnx_path = Path(effnet_onnx_path)
        self.mobilenet_onnx_path = Path(mobilenet_onnx_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Ensemble config not found: {self.config_path}")
        if not self.effnet_onnx_path.exists():
            raise FileNotFoundError(f"EfficientNet ONNX model not found: {self.effnet_onnx_path}")
        if not self.mobilenet_onnx_path.exists():
            raise FileNotFoundError(f"MobileNet ONNX model not found: {self.mobilenet_onnx_path}")

        # Load ensemble config
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.class_to_idx: Dict[str, int] = self.config["class_to_idx"]
        self.class_to_idx_reverse: Dict[str, str] = {
            str(k): v for k, v in self.config["class_to_idx_reverse"].items()
        }
        self.num_classes: int = len(self.class_to_idx)
        self.combination_method: str = self.config["combination_method"]
        self.weights = self.config.get("weights")
        self.low_support_classes = set(self.config.get("low_support_classes", []))

        # Initialize ONNX runtime sessions
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.effnet_session = ort.InferenceSession(
            str(self.effnet_onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.mobilenet_session = ort.InferenceSession(
            str(self.mobilenet_onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        self.effnet_input_name = self.effnet_session.get_inputs()[0].name
        self.mobilenet_input_name = self.mobilenet_session.get_inputs()[0].name

        # Startup safety check
        self._validate_models_on_startup()

    def _validate_models_on_startup(self):
        """
        Validates model output shapes against class count in config.
        Raises RuntimeError immediately if there is any mismatch to prevent silent corrupt inference.
        """
        effnet_out_shape = self.effnet_session.get_outputs()[0].shape
        mobilenet_out_shape = self.mobilenet_session.get_outputs()[0].shape

        effnet_classes = effnet_out_shape[-1]
        mobilenet_classes = mobilenet_out_shape[-1]

        if effnet_classes != self.num_classes:
            raise RuntimeError(
                f"[STARTUP SAFETY ERROR] EfficientNet output class count ({effnet_classes}) "
                f"does not match ensemble_config.json class count ({self.num_classes})! "
                "Refusing to start service in broken/misaligned state."
            )

        if mobilenet_classes != self.num_classes:
            raise RuntimeError(
                f"[STARTUP SAFETY ERROR] MobileNet output class count ({mobilenet_classes}) "
                f"does not match ensemble_config.json class count ({self.num_classes})! "
                "Refusing to start service in broken/misaligned state."
            )

    def predict(self, raw_image_input: Union[bytes, Image.Image]) -> Dict[str, Any]:
        """
        Runs dual ONNX forward passes and combines predictions according to config.
        """
        start_time = time.perf_counter()

        # 1. Preprocess image
        input_tensor = preprocess_fundus_image(raw_image_input)

        # 2. Run EfficientNet ONNX session
        effnet_logits = self.effnet_session.run(None, {self.effnet_input_name: input_tensor})[0]
        effnet_probs = softmax(effnet_logits, axis=1)[0]  # shape (num_classes,)

        # 3. Run MobileNet ONNX session
        mobilenet_logits = self.mobilenet_session.run(None, {self.mobilenet_input_name: input_tensor})[0]
        mobilenet_probs = softmax(mobilenet_logits, axis=1)[0]  # shape (num_classes,)

        # 4. Solo model predictions
        effnet_idx = int(np.argmax(effnet_probs))
        mobilenet_idx = int(np.argmax(mobilenet_probs))

        effnet_pred_class = self.class_to_idx_reverse[str(effnet_idx)]
        mobilenet_pred_class = self.class_to_idx_reverse[str(mobilenet_idx)]
        agree = bool(effnet_idx == mobilenet_idx)

        # 5. Combine probabilities based on configured combination_method
        if self.combination_method == "simple_average":
            combined_probs = (effnet_probs + mobilenet_probs) / 2.0
        elif self.combination_method == "weighted_average":
            w_eff = float(self.weights.get("efficientnet", 0.5))
            w_mob = float(self.weights.get("mobilenet", 0.5))
            combined_probs = (w_eff * effnet_probs) + (w_mob * mobilenet_probs)
        elif self.combination_method == "efficientnet_solo":
            combined_probs = effnet_probs
        elif self.combination_method == "mobilenet_solo":
            combined_probs = mobilenet_probs
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")

        # Ensure normalized sum to 1.0
        combined_probs = combined_probs / np.sum(combined_probs)

        pred_idx = int(np.argmax(combined_probs))
        predicted_class = self.class_to_idx_reverse[str(pred_idx)]
        confidence = float(combined_probs[pred_idx])

        # All class probabilities mapping
        class_probabilities = {
            self.class_to_idx_reverse[str(i)]: float(combined_probs[i])
            for i in range(self.num_classes)
        }

        is_low_support = bool(predicted_class in self.low_support_classes)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "combination_method": self.combination_method,
            "all_class_probabilities": class_probabilities,
            "solo_predictions": {
                "efficientnet_prediction": effnet_pred_class,
                "mobilenet_prediction": mobilenet_pred_class,
                "agree": agree,
            },
            "is_low_support_class": is_low_support,
            "inference_time_ms": elapsed_ms,
        }
