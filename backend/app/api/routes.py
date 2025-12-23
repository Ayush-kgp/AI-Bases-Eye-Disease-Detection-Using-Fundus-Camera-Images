from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from ..models.predictor import FundusPredictor
from ..utils.image_processing import process_image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
predictor = FundusPredictor()

@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint for predicting retinal conditions from fundus images.
    
    Args:
        file (UploadFile): The fundus image file to analyze
        
    Returns:
        JSONResponse: Contains the predicted condition
    """
    try:
        logger.info(f"Received prediction request for file: {file.filename}")
        
        # Process image and get prediction
        image = await process_image(file)
        prediction = predictor.predict(image)
        
        logger.info(f"Prediction made: {prediction}")
        return JSONResponse({"prediction": prediction})
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Prediction failed: {str(e)}"}
        )

@router.post("/gradcam/")
async def gradcam(file: UploadFile = File(...)):
    """
    Endpoint for generating Grad-CAM visualization for fundus images.
    
    Args:
        file (UploadFile): The fundus image file to analyze
        
    Returns:
        JSONResponse: Contains the prediction and Grad-CAM visualization
    """
    try:
        logger.info(f"Received Grad-CAM request for file: {file.filename}")
        
        # Process image and get Grad-CAM
        image = await process_image(file)
        prediction, gradcam_image = predictor.generate_gradcam(image)
        
        logger.info(f"Grad-CAM generated for prediction: {prediction}")
        return JSONResponse({
            "prediction": prediction,
            "gradcam_image": gradcam_image
        })
        
    except Exception as e:
        logger.error(f"Error in Grad-CAM generation: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Grad-CAM generation failed: {str(e)}"}
        )

@router.get("/")
async def home():
    """Health check endpoint."""
    return {"message": "Server is live. Use /predict or /gradcam via POST"} 