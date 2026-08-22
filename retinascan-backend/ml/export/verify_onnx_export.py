"""Verify ONNX export parity against PyTorch checkpoints on real test images."""

import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import onnxruntime as ort

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".ppm", ".pgm", ".webp"}


def build_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_pytorch_model(ckpt_path: Path, arch: str, num_classes: int = 10):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_macro_f1(preds, targets, num_classes=10):
    f1_scores = []
    for c in range(num_classes):
        tp = sum(1 for p, t in zip(preds, targets) if p == c and t == c)
        fp = sum(1 for p, t in zip(preds, targets) if p == c and t != c)
        fn = sum(1 for p, t in zip(preds, targets) if p != c and t == c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores))


def verify_model(
    model_name: str,
    arch: str,
    ckpt_path: Path,
    onnx_path: Path,
    test_loader: DataLoader,
    baseline_macro_f1: float,
    atol: float = 1e-4,
):
    print(f"\n{'='*70}")
    print(f"VERIFYING ONNX EXPORT: {model_name} ({arch})")
    print(f"{'='*70}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"ONNX Model: {onnx_path}")

    # Load PyTorch model
    py_model = load_pytorch_model(ckpt_path, arch)

    # Load ONNX Runtime session
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name

    max_diff = 0.0
    all_close_matches = True
    onnx_preds = []
    pytorch_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            # PyTorch inference
            py_logits = py_model(images).numpy()
            py_probs = torch.softmax(torch.from_numpy(py_logits), dim=1).numpy()

            # ONNX Runtime inference
            onnx_inputs = {input_name: images.numpy()}
            onnx_logits = ort_session.run(None, onnx_inputs)[0]
            onnx_probs = torch.softmax(torch.from_numpy(onnx_logits), dim=1).numpy()

            # Compute numerical difference on logits and probs
            batch_max_diff = float(np.max(np.abs(py_probs - onnx_probs)))
            if batch_max_diff > max_diff:
                max_diff = batch_max_diff

            if not np.allclose(py_probs, onnx_probs, atol=atol):
                all_close_matches = False

            onnx_preds.extend(np.argmax(onnx_probs, axis=1).tolist())
            pytorch_preds.extend(np.argmax(py_probs, axis=1).tolist())
            all_targets.extend(targets.tolist())

    onnx_macro_f1 = compute_macro_f1(onnx_preds, all_targets)
    pytorch_macro_f1 = compute_macro_f1(pytorch_preds, all_targets)

    print(f"Total test samples evaluated : {len(all_targets)}")
    print(f"Max absolute prob difference  : {max_diff:.6e} (tolerance atol={atol})")
    print(f"Numerical parity check (allclose): {'PASS' if all_close_matches else 'FAIL'}")
    print(f"PyTorch test macro-F1        : {pytorch_macro_f1:.4f}")
    print(f"ONNX Runtime test macro-F1   : {onnx_macro_f1:.4f}")
    print(f"Baseline recorded macro-F1   : {baseline_macro_f1:.4f}")
    print(f"Macro-F1 difference (ONNX vs Baseline): {abs(onnx_macro_f1 - baseline_macro_f1):.6f}")

    verdict = all_close_matches and (abs(onnx_macro_f1 - baseline_macro_f1) < 1e-3)

    if verdict:
        print(f"\n>>> VERDICT: PASS [OK]")
    else:
        print(f"\n>>> VERDICT: FAIL [MISMATCH]")

    return verdict


def main():
    root_dir = Path(__file__).resolve().parent.parent.parent
    workspace_root = root_dir.parent
    artifacts_dir = workspace_root / "training-artifacts"
    test_images_dir = artifacts_dir / "test-images"
    eval_json_path = artifacts_dir / "eval_baseline_results.json"

    if not test_images_dir.exists():
        print(f"[ERROR] Test images directory not found: {test_images_dir}")
        sys.exit(1)

    with open(eval_json_path, "r", encoding="utf-8") as f:
        baseline_results = json.load(f)

    test_dataset = datasets.ImageFolder(
        str(test_images_dir),
        transform=build_eval_transform(),
        is_valid_file=lambda p: Path(p).suffix.lower() in IMG_EXTENSIONS,
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    models_to_verify = [
        {
            "name": "EfficientNet-B0",
            "arch": "efficientnet_b0",
            "ckpt": artifacts_dir / "efficientnet_unfrozen_best.pth",
            "onnx": root_dir / "ml" / "models" / "efficientnet.onnx",
            "baseline_f1": baseline_results["efficientnet_unfrozen"]["test_macro_f1"],
        },
        {
            "name": "MobileNet-V2",
            "arch": "mobilenet_v2",
            "ckpt": artifacts_dir / "mobilenet_unfrozen_best.pth",
            "onnx": root_dir / "ml" / "models" / "mobilenet.onnx",
            "baseline_f1": baseline_results["mobilenet_unfrozen"]["test_macro_f1"],
        },
    ]

    all_passed = True
    for m in models_to_verify:
        passed = verify_model(
            model_name=m["name"],
            arch=m["arch"],
            ckpt_path=m["ckpt"],
            onnx_path=m["onnx"],
            test_loader=test_loader,
            baseline_macro_f1=m["baseline_f1"],
        )
        if not passed:
            all_passed = False

    print(f"\n{'='*70}")
    if all_passed:
        print("OVERALL ONNX VERIFICATION GATE: PASS (Both models verified)")
        print(f"{'='*70}")
        sys.exit(0)
    else:
        print("OVERALL ONNX VERIFICATION GATE: FAIL")
        print(f"{'='*70}")
        sys.exit(1)


if __name__ == "__main__":
    main()
