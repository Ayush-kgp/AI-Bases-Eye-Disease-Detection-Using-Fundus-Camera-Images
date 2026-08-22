"""Generate Grad-CAM heatmaps for sample Glaucoma and Diabetic Retinopathy test images."""

import os
import sys
from pathlib import Path
from PIL import Image

# Ensure retinascan-backend root is on sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.services.gradcam import GradCAMService

def main():
    workspace_root = root_dir.parent
    artifacts_dir = workspace_root / "training-artifacts"
    test_dir = artifacts_dir / "test-images"
    output_dir = root_dir / "docs" / "gradcam_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    effnet_ckpt = artifacts_dir / "efficientnet_unfrozen_best.pth"
    mobilenet_ckpt = artifacts_dir / "mobilenet_unfrozen_best.pth"

    print("Initializing Grad-CAM Service...")
    gcam_service = GradCAMService(
        effnet_ckpt_path=effnet_ckpt,
        mobilenet_ckpt_path=mobilenet_ckpt,
        num_classes=10,
    )

    targets = [
        ("Glaucoma", 3),
        ("Diabetic Retinopathy", 1),
    ]

    print(f"\nSaving Grad-CAM visualization samples to: {output_dir}\n")

    for class_name, class_idx in targets:
        class_folder = test_dir / class_name
        if not class_folder.exists():
            print(f"[WARN] Folder not found: {class_folder}")
            continue

        images = [
            f for f in sorted(os.listdir(class_folder))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ][:3]  # Take 3 sample images

        print(f"=== Class: {class_name} (target class index: {class_idx}) ===")
        for i, img_name in enumerate(images, 1):
            img_path = class_folder / img_name
            with open(img_path, "rb") as f:
                img_bytes = f.read()

            # Generate EfficientNet Grad-CAM
            _, eff_overlay = gcam_service.generate_gradcam_overlay(
                img_bytes,
                target_class_idx=class_idx,
                model_name="efficientnet",
                alpha=0.4,
            )
            eff_out_path = output_dir / f"{class_name.lower().replace(' ', '_')}_sample{i}_effnet.png"
            Image.fromarray(eff_overlay).save(eff_out_path)

            # Generate MobileNet Grad-CAM
            _, mob_overlay = gcam_service.generate_gradcam_overlay(
                img_bytes,
                target_class_idx=class_idx,
                model_name="mobilenet",
                alpha=0.4,
            )
            mob_out_path = output_dir / f"{class_name.lower().replace(' ', '_')}_sample{i}_mobilenet.png"
            Image.fromarray(mob_overlay).save(mob_out_path)

            print(f"  Sample {i} [{img_name}]:")
            print(f"    -> EfficientNet Grad-CAM: {eff_out_path.name}")
            print(f"    -> MobileNet Grad-CAM   : {mob_out_path.name}")

    print(f"\n[OK] Generated all Grad-CAM sample images successfully in {output_dir}")


if __name__ == "__main__":
    main()
