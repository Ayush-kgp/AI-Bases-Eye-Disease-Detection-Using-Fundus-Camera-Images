"""Pydantic schemas defining the API response contracts."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PredictionDetail(BaseModel):
    disease: str = Field(..., description="Predicted retinal disease name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence probability for predicted class")
    combination_method: str = Field(..., description="Combination method used (e.g. simple_average)")
    class_probabilities: Dict[str, float] = Field(
        ..., description="Full probability distribution across all classes"
    )


class ModelAgreement(BaseModel):
    efficientnet_prediction: str = Field(..., description="Solo prediction from EfficientNet-B0")
    mobilenet_prediction: str = Field(..., description="Solo prediction from MobileNet-V2")
    agree: bool = Field(..., description="True if both solo backbones predicted the same class")


class ReliabilityFlag(BaseModel):
    is_low_support_class: bool = Field(
        ..., description="True if class has fewer than 25 test samples in evaluation"
    )
    note: Optional[str] = Field(
        None, description="Clinical/statistical reliability advisory note if applicable"
    )


class Explainability(BaseModel):
    gradcam_overlay_base64: str = Field(
        ..., description="Data URI base64-encoded PNG of the Grad-CAM heatmap overlaid on original image"
    )


class PredictionMeta(BaseModel):
    inference_time_ms: float = Field(..., description="Total inference and heatmap generation latency in ms")
    model_version: str = Field(default="1.0.0", description="Deployed model artifacts version")


class PredictionResponse(BaseModel):
    prediction: PredictionDetail
    model_agreement: ModelAgreement
    reliability_flag: ReliabilityFlag
    explainability: Explainability
    meta: PredictionMeta
    disclaimer: str = Field(
        default="Research prototype — not a certified diagnostic tool.",
        description="Regulatory medical disclaimer",
    )


class ModelInfoResponse(BaseModel):
    combination_method: str
    weights: Optional[Dict[str, float]] = None
    solo_test_macro_f1: Dict[str, float]
    ensemble_test_macro_f1: Optional[float] = None
    low_support_classes: List[str]
    class_to_idx: Dict[str, int]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str = "1.0.0"
