"""Export trained EfficientNet-B0 PyTorch checkpoint to ONNX format."""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

def export_efficientnet(
    checkpoint_path: str,
    output_path: str,
    num_classes: int = 10,
    opset_version: int = 17,
):
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    ckpt_num_classes = ckpt.get("num_classes", num_classes)
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, ckpt_num_classes)

    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)

    # Critical: eval mode ensures BatchNorm and Dropout behave deterministically
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    print(f"Exporting to ONNX at: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,
    )
    print(f"[OK] EfficientNet exported successfully -> {output_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    default_ckpt = base_dir.parent / "training-artifacts" / "efficientnet_unfrozen_best.pth"
    default_out = base_dir / "ml" / "models" / "efficientnet.onnx"

    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else str(default_ckpt)
    out_path = sys.argv[2] if len(sys.argv) > 2 else str(default_out)

    export_efficientnet(ckpt_path, out_path)
