from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io
from ..models.predictor import FundusPredictor
import logging

logger = logging.getLogger(__name__)
router = APIRouter()
predictor = FundusPredictor()

@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Endpoint for fundus image prediction."""
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Make prediction
        prediction = predictor.predict(image)
        
        return {"prediction": prediction}
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict-with-gradcam")
async def predict_with_gradcam(file: UploadFile = File(...)):
    """Endpoint for fundus image prediction with Grad-CAM visualization."""
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Get prediction and Grad-CAM
        prediction, gradcam_image = predictor.generate_gradcam(image)
        
        return {
            "prediction": prediction,
            "gradcam_image": gradcam_image
        }
        
    except Exception as e:
        logger.error(f"Error in prediction with Grad-CAM endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 