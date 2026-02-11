"""
FastAPI service for Cats vs Dogs classification.
Includes health check, prediction endpoints, and basic monitoring.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Optional
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import torch

from src.inference import ModelInference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Cats vs Dogs Classifier API",
    description="Binary image classification service for cats and dogs",
    version="1.0.0"
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'prediction_requests_total',
    'Total number of prediction requests',
    ['endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'prediction_request_latency_seconds',
    'Prediction request latency in seconds',
    ['endpoint']
)
PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total number of predictions by class',
    ['predicted_class']
)

# Global model instance
model_inference: Optional[ModelInference] = None
startup_time = time.time()


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    uptime_seconds: float
    device: str


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model_inference
    
    logger.info("Starting up API service...")
    
    # Load model
    model_path = "models/best_model.pth"
    
    if not Path(model_path).exists():
        logger.error(f"Model file not found at {model_path}")
        raise RuntimeError(f"Model file not found: {model_path}")
    
    try:
        logger.info(f"Loading model from {model_path}")
        model_inference = ModelInference(model_path)
        logger.info(f"✓ Model loaded successfully on device: {model_inference.device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Cats vs Dogs Classifier API",
        "version": "1.0.0",
        "health_endpoint": "/health",
        "predict_endpoint": "/predict",
        "metrics_endpoint": "/metrics"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status of the service
    """
    uptime = time.time() - startup_time
    
    health_status = HealthResponse(
        status="healthy" if model_inference is not None else "unhealthy",
        model_loaded=model_inference is not None,
        uptime_seconds=round(uptime, 2),
        device=model_inference.device if model_inference else "unknown"
    )
    
    # Log health check
    logger.info(f"Health check: {health_status.status}")
    REQUEST_COUNT.labels(endpoint='health', status='success').inc()
    
    return health_status


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict image class.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction results with probabilities
    """
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            REQUEST_COUNT.labels(endpoint='predict', status='error').inc()
            raise HTTPException(
                status_code=400,
                detail=f"File must be an image. Got: {file.content_type}"
            )
        
        # Read image bytes
        image_bytes = await file.read()
        
        # Check model is loaded
        if model_inference is None:
            REQUEST_COUNT.labels(endpoint='predict', status='error').inc()
            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )
        
        # Make prediction
        logger.info(f"Processing prediction request for file: {file.filename}")
        result = model_inference.predict_from_bytes(image_bytes)
        
        # Calculate latency
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        # Update metrics
        REQUEST_COUNT.labels(endpoint='predict', status='success').inc()
        REQUEST_LATENCY.labels(endpoint='predict').observe(time.time() - start_time)
        PREDICTION_COUNT.labels(predicted_class=result['predicted_class']).inc()
        
        # Prepare response
        response = PredictionResponse(
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            probabilities={
                'cats': result['cats'],
                'dogs': result['dogs']
            },
            latency_ms=round(latency, 2)
        )
        
        # Log prediction
        logger.info(
            f"Prediction: {response.predicted_class} "
            f"(confidence: {response.confidence:.2%}, "
            f"latency: {response.latency_ms:.2f}ms)"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        REQUEST_COUNT.labels(endpoint='predict', status='error').inc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns:
        Prometheus metrics in text format
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/info")
async def model_info():
    """
    Get model information.
    
    Returns:
        Model metadata and configuration
    """
    if model_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "CatsDogsClassifier",
        "classes": model_inference.class_names,
        "num_classes": len(model_inference.class_names),
        "input_size": "224x224 RGB",
        "device": model_inference.device,
        "framework": "PyTorch"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
