"""
FastAPI Model Serving with Monitoring
Production-ready API for model inference
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import json
import logging
from prometheus_fastapi_instrumentator import Instrumentator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="ML Model API",
    description="Production ML Model Serving with Monitoring",
    version="1.0.0"
)

# Add Prometheus monitoring
Instrumentator().instrument(app).expose(app)


class PredictionInput(BaseModel):
    """Input schema for predictions"""
    features: List[float] = Field(..., description="Feature values for prediction")
    request_id: Optional[str] = Field(None, description="Optional request ID for tracking")


class PredictionOutput(BaseModel):
    """Output schema for predictions"""
    prediction: int
    probability: float
    request_id: Optional[str]
    timestamp: str
    model_version: str


class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions"""
    data: List[List[float]] = Field(..., description="Batch of feature vectors")


class ModelService:
    """Model serving service"""
    
    def __init__(self, model_path: str = "models/model.pkl", scaler_path: str = "models/scaler.pkl"):
        """Initialize model service"""
        self.model = None
        self.scaler = None
        self.model_version = "1.0.0"
        self.load_model(model_path, scaler_path)
        
    def load_model(self, model_path: str, scaler_path: str):
        """Load model and scaler"""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess(self, features: np.ndarray) -> np.ndarray:
        """Preprocess features"""
        return self.scaler.transform(features)
    
    def predict(self, features: np.ndarray) -> tuple:
        """Make prediction"""
        features_scaled = self.preprocess(features)
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        return int(prediction), float(max(probability))
    
    def batch_predict(self, features: np.ndarray) -> tuple:
        """Make batch predictions"""
        features_scaled = self.preprocess(features)
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        return predictions.tolist(), probabilities.tolist()


# Initialize model service
model_service = ModelService()


# Prediction logger
class PredictionLogger:
    """Log predictions for monitoring"""
    
    def __init__(self, log_file: str = "logs/predictions.jsonl"):
        self.log_file = log_file
        
    def log(self, data: dict):
        """Log prediction data"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            logger.error(f"Error logging prediction: {str(e)}")


prediction_logger = PredictionLogger()


@app.on_event("startup")
async def startup_event():
    """Startup event"""
    logger.info("Starting ML API...")
    logger.info(f"Model version: {model_service.model_version}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ML Model API",
        "version": model_service.model_version,
        "status": "healthy"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_service.model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput, background_tasks: BackgroundTasks):
    """
    Make a single prediction
    
    - **features**: List of feature values
    - **request_id**: Optional request ID for tracking
    """
    try:
        # Convert to numpy array
        features = np.array(input_data.features).reshape(1, -1)
        
        # Make prediction
        prediction, probability = model_service.predict(features)
        
        # Prepare response
        response = PredictionOutput(
            prediction=prediction,
            probability=probability,
            request_id=input_data.request_id,
            timestamp=datetime.now().isoformat(),
            model_version=model_service.model_version
        )
        
        # Log prediction in background
        log_data = {
            "timestamp": response.timestamp,
            "request_id": input_data.request_id,
            "features": input_data.features,
            "prediction": prediction,
            "probability": probability,
            "model_version": model_service.model_version
        }
        background_tasks.add_task(prediction_logger.log, log_data)
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch-predict")
async def batch_predict(input_data: BatchPredictionInput):
    """
    Make batch predictions
    
    - **data**: List of feature vectors
    """
    try:
        # Convert to numpy array
        features = np.array(input_data.data)
        
        # Make predictions
        predictions, probabilities = model_service.batch_predict(features)
        
        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "count": len(predictions),
            "timestamp": datetime.now().isoformat(),
            "model_version": model_service.model_version
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "model_version": model_service.model_version,
        "model_type": type(model_service.model).__name__,
        "feature_count": model_service.scaler.n_features_in_ if model_service.scaler else None,
        "model_params": model_service.model.get_params() if hasattr(model_service.model, 'get_params') else None
    }


@app.post("/reload-model")
async def reload_model(model_path: str = "models/model.pkl", scaler_path: str = "models/scaler.pkl"):
    """Reload model (admin endpoint)"""
    try:
        model_service.load_model(model_path, scaler_path)
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
