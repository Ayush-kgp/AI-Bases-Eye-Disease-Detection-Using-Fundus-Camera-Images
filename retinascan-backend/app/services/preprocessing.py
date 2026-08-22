"""Image preprocessing replicating the exact training and evaluation pipeline."""

import io
from typing import Union
import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def preprocess_fundus_image(image_input: Union[bytes, Image.Image]) -> np.ndarray:
    """
    Replicates torchvision transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    Args:
        image_input: Raw image bytes or PIL.Image.Image instance

    Returns:
        np.ndarray of shape (1, 3, 224, 224), float32 normalized tensor
    """
    if isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    else:
        raise TypeError(f"Expected bytes or PIL Image, got {type(image_input)}")

    # Convert to RGB (handles RGBA, grayscale, paletted images)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Resize to 224x224 using Bilinear interpolation (torchvision default for Resize((224, 224)))
    resized_img = pil_img.resize((224, 224), Image.BILINEAR)

    # Convert to numpy array: (224, 224, 3) in uint8 [0, 255]
    img_np = np.array(resized_img, dtype=np.float32) / 255.0

    # Permute dimensions to (1, 3, 224, 224) - NCHW
    img_nchw = np.transpose(img_np, (2, 0, 1))[np.newaxis, :, :, :]

    # Normalize with ImageNet mean and std
    normalized_tensor = (img_nchw - IMAGENET_MEAN) / IMAGENET_STD

    return normalized_tensor.astype(np.float32)
