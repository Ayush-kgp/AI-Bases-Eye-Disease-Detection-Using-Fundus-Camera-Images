"""Prediction and explainability API route."""

import time
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.schemas.prediction import (
    Explainability,
    ModelAgreement,
    PredictionDetail,
    PredictionMeta,
    PredictionResponse,
    ReliabilityFlag,
)
from app.utils.validators import validate_image_file

router = APIRouter(tags=["Inference"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Retinal Disease with Grad-CAM Explainability",
    description="Upload a retinal fundus image for dual-model classification and Grad-CAM heatmap visualization.",
)
async def predict_fundus_image(
    request: Request,
    file: UploadFile = File(..., description="Fundus camera image (JPEG/PNG/WEBP)"),
):
    total_start = time.perf_counter()

    # 1. Retrieve services from application state
    inference_service = getattr(request.app.state, "inference_service", None)
    gradcam_service = getattr(request.app.state, "gradcam_service", None)

    if inference_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service is not initialized.",
        )

    # 2. Validate input image
    image_bytes, pil_img = await validate_image_file(file)

    # 3. Run dual ONNX ensemble inference
    inf_result = inference_service.predict(image_bytes)
    predicted_disease = inf_result["predicted_class"]
    pred_idx = inference_service.class_to_idx[predicted_disease]

    # 4. Run Grad-CAM explainability
    overlay_data_uri = ""
    if gradcam_service is not None:
        try:
            overlay_data_uri, _ = gradcam_service.generate_gradcam_overlay(
                image_bytes,
                target_class_idx=pred_idx,
                model_name="efficientnet",
                alpha=0.4,
            )
        except Exception as e:
            # Fallback if Grad-CAM encounter an issue
            overlay_data_uri = f"data:text/plain;base64,{str(e)}"

    total_latency_ms = (time.perf_counter() - total_start) * 1000.0

    # 5. Build reliability note for low support classes
    note = None
    if inf_result["is_low_support_class"]:
        note = (
            f"Class '{predicted_disease}' has limited test samples (<25) in validation data. "
            "Interpret prediction with clinical discretion."
        )

    return PredictionResponse(
        prediction=PredictionDetail(
            disease=predicted_disease,
            confidence=inf_result["confidence"],
            combination_method=inf_result["combination_method"],
            class_probabilities=inf_result["all_class_probabilities"],
        ),
        model_agreement=ModelAgreement(
            efficientnet_prediction=inf_result["solo_predictions"]["efficientnet_prediction"],
            mobilenet_prediction=inf_result["solo_predictions"]["mobilenet_prediction"],
            agree=inf_result["solo_predictions"]["agree"],
        ),
        reliability_flag=ReliabilityFlag(
            is_low_support_class=inf_result["is_low_support_class"],
            note=note,
        ),
        explainability=Explainability(
            gradcam_overlay_base64=overlay_data_uri,
        ),
        meta=PredictionMeta(
            inference_time_ms=round(total_latency_ms, 2),
            model_version="1.0.0",
        ),
        disclaimer="Research prototype — not a certified diagnostic tool.",
    )
