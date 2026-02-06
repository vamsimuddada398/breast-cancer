"""
FastAPI Backend for Breast Cancer Detection
Provides REST API for mammogram image upload and cancer detection inference
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io
import base64
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Breast Cancer Detection API",
    description="AI-powered mammography analysis for breast cancer detection and risk assessment",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Models ====================

class PredictionResponse(BaseModel):
    """Response model for cancer detection predictions"""
    prediction: str  # 'benign' or 'malignant'
    confidence: float
    probabilities: Dict[str, float]
    risk_score: float
    risk_category: str
    recommendations: List[str]
    timestamp: str
    model_used: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    models_loaded: bool


class SegmentationResponse(BaseModel):
    """Segmentation response"""
    segmentation_mask: str  # Base64 encoded PNG
    confidence: float
    timestamp: str
    model_used: str


class ModelsResponse(BaseModel):
    """Models information response"""
    models: Dict[str, Dict[str, str]]
    default_model: str
    segmentation_available: bool


# ==================== Global Variables ====================

# Model paths (update with your actual model paths)
MODEL_PATHS = {
    'efficientnet': "../trained_models/efficientnet_final_savedmodel",
    'efficientnet_h5': "./models/efficientnet_final.h5",
    'resnet': "./models/resnet_model.h5",
    'densenet': "./models/densenet_model.h5"
}

UNET_MODEL_PATH = "./models/unet_segmentation.h5"

# Available models
models = {}
segmentation_model = None

# Load classification models
for model_name, model_path in MODEL_PATHS.items():
    try:
        print(f"Loading {model_name} model...")
        if model_name == 'efficientnet':
            # Load SavedModel format
            model = tf.saved_model.load(model_path)
            model = model.signatures['serving_default']
        else:
            # Load Keras H5 format
            model = tf.keras.models.load_model(model_path, compile=False)

        models[model_name] = model
        logger.info(f"✓ {model_name} model loaded successfully")
        print(f"✓ {model_name} model loaded successfully")
    except Exception as e:
        logger.warning(f"{model_name} model not loaded: {e}")
        print(f"⚠ {model_name} model not loaded: {e}")

# Load segmentation model (U-Net)
try:
    print("Loading U-Net segmentation model...")
    segmentation_model = tf.keras.models.load_model(
        UNET_MODEL_PATH,
        compile=False
    )
    logger.info("✓ U-Net segmentation model loaded successfully")
    print("✓ U-Net segmentation model loaded successfully")
except Exception as e:
    logger.warning(f"U-Net segmentation model not loaded: {e}")
    segmentation_model = None
    print(f"⚠ U-Net segmentation model not loaded: {e}")

# Default model
default_model_name = 'efficientnet' if 'efficientnet' in models else list(models.keys())[0] if models else None


# ==================== Helper Functions ====================

def preprocess_image(image_bytes: bytes, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocess uploaded image for model inference
    
    Args:
        image_bytes: Raw image bytes
        target_size: Target dimensions
    
    Returns:
        Preprocessed image array
    """
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_array = clahe.apply(img_array)
    
    # Resize
    img_array = cv2.resize(img_array, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert to RGB (for model input)
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Normalize
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch


def calculate_risk_score(probabilities: Dict[str, float], 
                        confidence: float) -> tuple:
    """
    Calculate overall risk score and category
    
    Args:
        probabilities: Class probabilities
        confidence: Prediction confidence
    
    Returns:
        Tuple of (risk_score, risk_category)
    """
    malignant_prob = probabilities['malignant']
    
    # Risk score is weighted by malignant probability and confidence
    risk_score = malignant_prob * confidence
    
    # Categorize risk
    if risk_score < 0.2:
        risk_category = "Low Risk"
    elif risk_score < 0.5:
        risk_category = "Moderate Risk"
    elif risk_score < 0.75:
        risk_category = "High Risk"
    else:
        risk_category = "Very High Risk"
    
    return risk_score, risk_category


def generate_recommendations(prediction: str, 
                            risk_category: str,
                            confidence: float) -> List[str]:
    """
    Generate clinical recommendations based on prediction
    
    Args:
        prediction: 'benign' or 'malignant'
        risk_category: Risk category
        confidence: Prediction confidence
    
    Returns:
        List of recommendations
    """
    recommendations = []
    
    if prediction == 'malignant' or risk_category in ['High Risk', 'Very High Risk']:
        recommendations.extend([
            "Immediate follow-up with radiologist recommended",
            "Consider additional imaging (ultrasound or MRI)",
            "Biopsy consultation may be necessary",
            "Schedule appointment within 1-2 weeks"
        ])
    elif risk_category == 'Moderate Risk':
        recommendations.extend([
            "Follow-up mammogram in 6 months recommended",
            "Continue regular screening schedule",
            "Discuss findings with your healthcare provider"
        ])
    else:
        recommendations.extend([
            "Continue routine annual screening",
            "Maintain healthy lifestyle practices",
            "Report any changes in breast tissue immediately"
        ])
    
    if confidence < 0.7:
        recommendations.append(
            "Note: Prediction confidence is moderate. Additional review recommended."
        )
    
    return recommendations


def segment_lesion(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Segment potential lesions using U-Net
    
    Args:
        image_bytes: Raw image bytes
    
    Returns:
        Segmentation mask or None
    """
    if segmentation_model is None:
        return None
    
    try:
        # Preprocess for segmentation (256x256, grayscale)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        img_array = np.array(image)
        img_array = cv2.resize(img_array, (256, 256))
        img_normalized = img_array.astype(np.float32) / 255.0
        img_batch = np.expand_dims(np.expand_dims(img_normalized, axis=-1), axis=0)
        
        # Predict segmentation mask
        mask = segmentation_model.predict(img_batch, verbose=0)[0]
        
        return mask
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        return None


# ==================== API Endpoints ====================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="online",
        timestamp=datetime.now().isoformat(),
        models_loaded=len(models) > 0
    )


@app.get("/health", response_model=HealthResponse)

async def segment_lesion_endpoint(
    file: UploadFile = File(...),
    model_name: str = "efficientnet"
):
    """
    Segment lesions in mammogram image using U-Net

    Args:
        file: Uploaded mammogram image (JPEG, PNG)
        model_name: Model used for context (efficientnet, resnet, densenet)

    Returns:
        Segmentation mask and confidence
    """
    if segmentation_model is None:
        raise HTTPException(
            status_code=503,
            detail="Segmentation model not loaded. Please contact administrator."
        )

    try:
        # Read and validate image
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Perform segmentation
        mask = segment_lesion(image_bytes)
        if mask is None:
            raise HTTPException(
                status_code=500,
                detail="Segmentation failed"
            )

        # Calculate confidence (mean of mask values)
        seg_confidence = float(np.mean(mask))

        # Convert mask to base64
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        buffer = io.BytesIO()
        mask_pil.save(buffer, format='PNG')
        mask_b64 = base64.b64encode(buffer.getvalue()).decode()

        logger.info(f"Segmentation completed with confidence: {seg_confidence:.2%}")

        return SegmentationResponse(
            segmentation_mask=mask_b64,
            confidence=seg_confidence,
            timestamp=datetime.now().isoformat(),
            model_used=model_name.title()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.get("/api/models")
async def get_available_models():
    """Get list of available models"""
    return {
        "available_models": list(models.keys()),
        "default_model": default_model_name,
        "segmentation_available": segmentation_model is not None,
        "total_models": len(models)
    }


@app.get("/api/models/{model_name}/info")
async def get_model_info(model_name: str):
    """Get information about a specific model"""
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    model = models[model_name]

    # Try to get model summary
    try:
        if hasattr(model, 'summary'):
            # Keras model
            summary_lines = []
            model.summary(print_fn=lambda x: summary_lines.append(x))
            summary = '\n'.join(summary_lines)
        else:
            # SavedModel
            summary = "TensorFlow SavedModel (signature-based inference)"
    except:
        summary = "Model summary not available"

    return {
        "model_name": model_name,
        "type": "Keras" if hasattr(model, 'summary') else "SavedModel",
        "summary": summary,
        "loaded": True
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint"""
    return HealthResponse(
        status="healthy" if classification_model is not None else "degraded",
        timestamp=datetime.now().isoformat(),
        models_loaded=classification_model is not None
    )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_cancer(
    file: UploadFile = File(...),
    model_name: str = "efficientnet"
):
    """
    Upload mammogram image and get cancer detection prediction

    Args:
        file: Uploaded mammogram image (JPEG, PNG)
        model_name: Model to use for prediction (efficientnet, resnet, densenet)

    Returns:
        Prediction results with probabilities and recommendations
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)"
        )

    # Check if model is loaded
    if model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not available. Available models: {list(models.keys())}"
        )

    if not models:
        raise HTTPException(
            status_code=503,
            detail="No models loaded. Please contact administrator."
        )

    try:
        # Read image bytes
        image_bytes = await file.read()

        # Preprocess image
        processed_image = preprocess_image(image_bytes)

        # Get selected model
        selected_model = models[model_name]

        # Make prediction based on model type
        if model_name == 'efficientnet':
            # SavedModel format
            predictions = selected_model(keras_tensor_386=processed_image)['output_0'].numpy()[0]
            benign_prob = float(predictions[0])
            malignant_prob = float(predictions[1])
        else:
            # Keras model format
            predictions = selected_model.predict(processed_image, verbose=0)[0]
            benign_prob = float(predictions[0])
            malignant_prob = float(predictions[1])

        # Parse results
        predicted_class_idx = np.argmax([benign_prob, malignant_prob])
        predicted_class = 'benign' if predicted_class_idx == 0 else 'malignant'
        confidence = max(benign_prob, malignant_prob)

        probabilities = {
            'benign': benign_prob,
            'malignant': malignant_prob
        }
        
        # Calculate risk score
        risk_score, risk_category = calculate_risk_score(probabilities, confidence)
        
        # Generate recommendations
        recommendations = generate_recommendations(
            predicted_class, risk_category, confidence
        )
        
        # Optional: Segment lesions
        # segmentation_mask = segment_lesion(image_bytes)
        
        logger.info(
            f"Prediction: {predicted_class} "
            f"(confidence: {confidence:.2%}, risk: {risk_category})"
        )
        
        return PredictionResponse(
            prediction=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            risk_score=risk_score,
            risk_category=risk_category,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat(),
            model_used=model_name.title()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/api/batch-predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    """
    Batch prediction for multiple mammogram images
    
    Args:
        files: List of uploaded images
    
    Returns:
        List of predictions
    """
    if classification_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please contact administrator."
        )
    
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images allowed per batch"
        )
    
    results = []
    
    for file in files:
        try:
            image_bytes = await file.read()
            processed_image = preprocess_image(image_bytes)
            predictions = classification_model(keras_tensor_386=processed_image)['output_0'].numpy()[0]
            
            predicted_class_idx = np.argmax(predictions)
            predicted_class = 'benign' if predicted_class_idx == 0 else 'malignant'
            confidence = float(predictions[predicted_class_idx])
            
            probabilities = {
                'benign': float(predictions[0]),
                'malignant': float(predictions[1])
            }
            
            risk_score, risk_category = calculate_risk_score(probabilities, confidence)
            
            results.append({
                'filename': file.filename,
                'prediction': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities,
                'risk_score': risk_score,
                'risk_category': risk_category
            })
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return JSONResponse(content={'results': results})


@app.get("/api/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if classification_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # For SavedModel, return static info
        return {
            'model_name': 'EfficientNetB3',
            'input_shape': [None, 224, 224, 3],
            'output_shape': [None, 2],
            'total_parameters': 'Unknown (SavedModel)',
            'classes': ['benign', 'malignant'],
            'preprocessing': {
                'target_size': [224, 224],
                'normalization': '0-1',
                'clahe': True
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("Breast Cancer Detection API Server")
    print("="*60)
    print("Starting server on http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    # Run the server without reload
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)