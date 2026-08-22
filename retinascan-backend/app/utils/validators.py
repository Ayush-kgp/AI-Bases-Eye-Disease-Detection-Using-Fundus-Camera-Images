"""Image input validation utilities."""

import io
from typing import Tuple
from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/bmp", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_RESOLUTION = (64, 64)
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB max


async def validate_image_file(file: UploadFile) -> Tuple[bytes, Image.Image]:
    """
    Validates uploaded image file:
    1. Checks filename and content type.
    2. Checks file size.
    3. Verifies image integrity and decodability with PIL.
    4. Enforces minimum resolution requirements.

    Returns:
        Tuple of (image_bytes, PIL.Image)
    """
    # 1. Check filename extension if available
    filename = file.filename or ""
    extension = "." + filename.split(".")[-1].lower() if "." in filename else ""

    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_CONTENT_TYPES and extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{content_type or extension}'. Allowed types: JPEG, PNG, BMP, WEBP.",
        )

    # Read bytes
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    if not image_bytes or len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB.",
        )

    # 2. Decode and verify integrity with PIL
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.verify()  # Verifies file headers and structure
    except (UnidentifiedImageError, OSError, ValueError, Exception) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file is corrupted or not a valid decodable image: {str(e)}",
        )

    # Reopen after verify() because verify() invalidates the PIL image object
    pil_img = Image.open(io.BytesIO(image_bytes))

    # 3. Minimum resolution check
    width, height = pil_img.size
    min_w, min_h = MIN_RESOLUTION
    if width < min_w or height < min_h:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Image resolution ({width}x{height}) is below the required minimum "
                f"resolution of {min_w}x{min_h} pixels for retinal fundus analysis."
            ),
        )

    return image_bytes, pil_img
