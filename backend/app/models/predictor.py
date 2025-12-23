import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
import base64
from io import BytesIO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import logging
import os

logger = logging.getLogger(__name__)

class FundusPredictor:
    def __init__(self):
        """Initialize the FundusPredictor with model and preprocessing setup."""
        self.num_classes = 39
        self.class_names = [
            '0.0.Normal', '0.1.Tessellated fundus', '0.2.Large optic cup', '0.3.DR1', 
            '1.0.DR2', '1.1.DR3', '10.0.Possible glaucoma', '10.1.Optic atrophy', 
            '11.Severe hypertensive retinopathy', '12.Disc swelling and elevation', 
            '13.Dragged Disc', '14.Congenital disc abnormality', '15.0.Retinitis pigmentosa', 
            '15.1.Bietti crystalline dystrophy', '16.Peripheral retinal degeneration and break', 
            '17.Myelinated nerve fiber', '18.Vitreous particles', '19.Fundus neoplasm', 
            '2.0.BRVO', '2.1.CRVO', '20.Massive hard exudates', '21.Yellow-white spots-flecks', 
            '22.Cotton-wool spots', '23.Vessel tortuosity', '24.Chorioretinal atrophy-coloboma', 
            '25.Preretinal hemorrhage', '26.Fibrosis', '27.Laser Spots', '28.Silicon oil in eye', 
            '29.0.Blur fundus without PDR', '29.1.Blur fundus with suspected PDR', '3.RAO', 
            '4.Rhegmatogenous RD', '5.0.CSCR', '5.1.VKH disease', '6.Maculopathy', 
            '7.ERM', '8.MH', '9.Pathological myopia'
        ]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model = self._load_model()
        self.transform = self._setup_transforms()
        
    def _load_model(self):
        """Load and initialize the EfficientNet model."""
        try:
            # Get the absolute path to the models directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'models')
            model_path = os.path.join(models_dir, 'efficientnet_best.pth')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            model = models.efficientnet_b0(pretrained=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def _setup_transforms(self):
        """Setup image preprocessing transforms."""
        return transforms.Compose([
            transforms.Lambda(self._apply_clahe),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
        ])
        
    def _apply_clahe(self, pil_img):
        """Apply CLAHE enhancement to the image."""
        try:
            img = np.array(pil_img)
            img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
            img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
            return Image.fromarray(img_output)
        except Exception as e:
            logger.error(f"Error in CLAHE processing: {str(e)}")
            raise
            
    def predict(self, image):
        """Make prediction for the given image."""
        try:
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                pred = output.argmax(1).item()
                
            return self.class_names[pred]
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise
            
    def generate_gradcam(self, image):
        """Generate Grad-CAM visualization for the given image."""
        try:
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Setup Grad-CAM
            target_layers = [self.model.features[-1]]
            cam = GradCAM(model=self.model, target_layers=target_layers)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                pred = output.argmax(1).item()
            
            # Generate Grad-CAM
            grayscale_cam = cam(input_tensor=input_tensor, 
                              targets=[ClassifierOutputTarget(pred)])[0]
            
            # Prepare image for visualization
            rgb_img = self.transform(image).permute(1, 2, 0).cpu().numpy()
            rgb_img = np.clip(rgb_img, 0, 1)
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            # Convert to base64
            cam_pil = Image.fromarray(cam_image)
            buffer = BytesIO()
            cam_pil.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode()
            
            return self.class_names[pred], f"data:image/png;base64,{encoded}"
            
        except Exception as e:
            logger.error(f"Error in Grad-CAM generation: {str(e)}")
            raise 