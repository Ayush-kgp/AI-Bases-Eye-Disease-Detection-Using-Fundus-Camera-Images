"""Health check and model evaluation info routes."""

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.schemas.prediction import HealthResponse, ModelInfoResponse

router = APIRouter(tags=["System & Model Info"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Returns 200 if models are loaded and healthy, 503 otherwise.",
)
async def health_check(request: Request):
    inference_service = getattr(request.app.state, "inference_service", None)
    if inference_service is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "models_loaded": False, "version": "1.0.0"},
        )

    return HealthResponse(status="healthy", models_loaded=True, version="1.0.0")


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model Evaluation & Configuration Metadata",
    description="Returns full evaluation metrics, winning combination strategy, and class mappings.",
)
async def model_info(request: Request):
    inference_service = getattr(request.app.state, "inference_service", None)
    if inference_service is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Inference service not initialized"},
        )

    cfg = inference_service.config
    return ModelInfoResponse(
        combination_method=cfg["combination_method"],
        weights=cfg.get("weights"),
        solo_test_macro_f1=cfg["solo_test_macro_f1"],
        ensemble_test_macro_f1=cfg.get("ensemble_test_macro_f1"),
        low_support_classes=cfg.get("low_support_classes", []),
        class_to_idx=cfg["class_to_idx"],
    )
