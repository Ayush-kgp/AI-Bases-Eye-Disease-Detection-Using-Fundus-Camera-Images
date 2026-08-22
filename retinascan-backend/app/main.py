"""Main FastAPI application entrypoint for RetinaScan AI backend."""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routes.health import router as health_router
from app.routes.predict import router as predict_router
from app.services.gradcam import GradCAMService
from app.services.inference import EnsembleInferenceService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("retinascan")

BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = BASE_DIR.parent
ARTIFACTS_DIR = WORKSPACE_ROOT / "training-artifacts"

CONFIG_PATH = BASE_DIR / "ml" / "models" / "ensemble_config.json"
EFFNET_ONNX = BASE_DIR / "ml" / "models" / "efficientnet.onnx"
MOBILENET_ONNX = BASE_DIR / "ml" / "models" / "mobilenet.onnx"

EFFNET_PTH = ARTIFACTS_DIR / "efficientnet_unfrozen_best.pth"
MOBILENET_PTH = ARTIFACTS_DIR / "mobilenet_unfrozen_best.pth"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: initializes models once at startup
    and gracefully tears down on shutdown.
    """
    logger.info("Initializing RetinaScan AI backend services...")

    try:
        # Initialize ONNX inference service
        logger.info(f"Loading ONNX models from {BASE_DIR / 'ml' / 'models'}...")
        app.state.inference_service = EnsembleInferenceService(
            config_path=CONFIG_PATH,
            effnet_onnx_path=EFFNET_ONNX,
            mobilenet_onnx_path=MOBILENET_ONNX,
        )
        logger.info("[OK] Dual-model ONNX EnsembleInferenceService loaded successfully.")
    except Exception as e:
        logger.error(f"[FATAL] Failed to initialize EnsembleInferenceService: {e}", exc_info=True)
        app.state.inference_service = None

    try:
        # Initialize PyTorch Grad-CAM service
        if EFFNET_PTH.exists():
            logger.info("Loading PyTorch checkpoints for Grad-CAM explainability...")
            app.state.gradcam_service = GradCAMService(
                effnet_ckpt_path=EFFNET_PTH,
                mobilenet_ckpt_path=MOBILENET_PTH if MOBILENET_PTH.exists() else None,
                num_classes=10,
            )
            logger.info("[OK] GradCAMService initialized successfully.")
        else:
            logger.warning(f"Grad-CAM PyTorch checkpoints not found at {ARTIFACTS_DIR}. Grad-CAM disabled.")
            app.state.gradcam_service = None
    except Exception as e:
        logger.error(f"[WARN] Failed to initialize GradCAMService: {e}")
        app.state.gradcam_service = None

    yield

    logger.info("Shutting down RetinaScan AI backend services...")


app = FastAPI(
    title="RetinaScan AI API",
    description="Production-grade 10-Class Retinal Disease Detection using Dual-CNN Ensemble & Grad-CAM",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Inline request logging middleware logging method, path, status, and duration."""
    start_time = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        status_code = response.status_code if response is not None else 500
        logger.info(f"{request.method} {request.url.path} -> {status_code} ({duration_ms:.2f}ms)")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler preventing unhandled 500 stack trace leaks."""
    logger.error(f"Unhandled error processing {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred during processing. Please try again later.",
            "path": request.url.path,
        },
    )


# Include Routers
app.include_router(health_router)
app.include_router(predict_router)
