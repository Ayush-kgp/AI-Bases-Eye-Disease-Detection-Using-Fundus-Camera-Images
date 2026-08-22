"""Grad-CAM explainability service for PyTorch backbones (EfficientNet and MobileNet).

NOTE ON DESIGN:
Grad-CAM requires computational graph and gradient access to intermediate convolutional layers,
which ONNX Runtime inference sessions do not provide natively. Therefore, Grad-CAM loads the
PyTorch (.pth) model checkpoints directly in PyTorch eval mode with grad enabled specifically
during the visualization pass, while fast production inference is served via ONNX Runtime.
"""

import base64
import io
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models

from app.services.preprocessing import preprocess_fundus_image


class GradCAM:
    """Lightweight, self-contained Grad-CAM implementation."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generates a 2D float32 heatmap in range [0, 1].
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        logits = self.model(input_tensor)

        if target_class_idx is None:
            target_class_idx = int(logits.argmax(dim=1).item())

        score = logits[0, target_class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients/activations.")

        # Global average pooling on gradients -> channel weights alpha_k
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, H, W]

        # Apply ReLU
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    def remove_hooks(self):
        for h in self._handles:
            h.remove()


class GradCAMService:
    """Manages PyTorch models and Grad-CAM generation for both backbones."""

    def __init__(
        self,
        effnet_ckpt_path: Union[str, Path],
        mobilenet_ckpt_path: Optional[Union[str, Path]] = None,
        num_classes: int = 10,
    ):
        self.num_classes = num_classes
        self.effnet_model = self._load_model("efficientnet_b0", Path(effnet_ckpt_path))
        self.effnet_target_layer = self.effnet_model.features[-1]

        self.mobilenet_model = None
        self.mobilenet_target_layer = None
        if mobilenet_ckpt_path and Path(mobilenet_ckpt_path).exists():
            self.mobilenet_model = self._load_model("mobilenet_v2", Path(mobilenet_ckpt_path))
            # In MobileNet-V2, features[-1] is the final 1x1 Conv2dNormActivation layer
            self.mobilenet_target_layer = self.mobilenet_model.features[-1]

    def _load_model(self, arch: str, ckpt_path: Path) -> nn.Module:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"PyTorch checkpoint not found: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        num_classes = ckpt.get("num_classes", self.num_classes)

        if arch == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif arch == "mobilenet_v2":
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def generate_gradcam_overlay(
        self,
        raw_image_input: Union[bytes, Image.Image],
        target_class_idx: Optional[int] = None,
        model_name: str = "efficientnet",
        alpha: float = 0.4,
    ) -> Tuple[str, np.ndarray]:
        """
        Generates Grad-CAM heatmap overlay.

        Returns:
            Tuple of:
            - base64_data_uri: str ("data:image/png;base64,...")
            - overlay_rgb_uint8: np.ndarray (H, W, 3) in uint8 [0, 255]
        """
        # Load original image for visualization
        if isinstance(raw_image_input, bytes):
            orig_pil = Image.open(io.BytesIO(raw_image_input)).convert("RGB")
        else:
            orig_pil = raw_image_input.convert("RGB")

        # Resize original image to 224x224 for standard overlay alignment
        orig_resized = orig_pil.resize((224, 224), Image.BILINEAR)
        orig_np = np.array(orig_resized, dtype=np.float32) / 255.0  # RGB [0, 1]

        # Select model and target layer
        if model_name == "mobilenet" and self.mobilenet_model is not None:
            model = self.mobilenet_model
            target_layer = self.mobilenet_target_layer
        else:
            model = self.effnet_model
            target_layer = self.effnet_target_layer

        # Preprocess image tensor
        tensor_np = preprocess_fundus_image(raw_image_input)
        tensor = torch.from_numpy(tensor_np)

        # Run Grad-CAM
        gcam = GradCAM(model, target_layer)
        try:
            heatmap_2d = gcam.generate_heatmap(tensor, target_class_idx=target_class_idx)
        finally:
            gcam.remove_hooks()

        # Resize heatmap to match image shape
        heatmap_resized = cv2.resize(heatmap_2d, (224, 224))

        # Apply JET colormap: cv2 expects uint8 [0, 255] and returns BGR
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Alpha blend overlay: alpha * heatmap + (1 - alpha) * original
        overlay_float = (alpha * heatmap_rgb) + ((1.0 - alpha) * orig_np)
        overlay_float = np.clip(overlay_float, 0.0, 1.0)
        overlay_uint8 = np.uint8(255 * overlay_float)

        # Convert to Base64 PNG
        overlay_pil = Image.fromarray(overlay_uint8)
        buffer = io.BytesIO()
        overlay_pil.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        data_uri = f"data:image/png;base64,{encoded}"

        return data_uri, overlay_uint8
