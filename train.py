"""
MLOps Pipeline with MLflow, DVC, and Model Monitoring
A complete end-to-end ML pipeline with experiment tracking and versioning
"""

import os
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLOpsPipeline:
    """Complete MLOps Pipeline with tracking and versioning"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize pipeline with configuration"""
        self.config = self.load_config(config_path)
        self.model = None
        self.scaler = None
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def load_data(self, data_path: str) -> tuple:
        """Load and split data"""
        logger.info(f"Loading data from {data_path}")
        
        df = pd.read_csv(data_path)
        
        # Separate features and target
        X = df.drop(columns=[self.config['data']['target_column']])
        y = df[self.config['data']['target_column']]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y
        )
        
        logger.info(f"Data loaded: Train shape {X_train.shape}, Test shape {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_data(self, X_train, X_test):
        """Preprocess features"""
        logger.info("Preprocessing data...")
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Fit and transform training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Transform test data
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def train_model(self, X_train, y_train):
        """Train model with hyperparameter tuning"""
        logger.info("Training model with hyperparameter tuning...")
        
        # Get model parameters from config
        param_grid = self.config['model']['param_grid']
        
        # Initialize model
        base_model = RandomForestClassifier(random_state=42)
        
        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit model
        grid_search.fit(X_train, y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_, grid_search.best_score_
    
    def evaluate_model(self, X_test, y_test) -> dict:
        """Evaluate model and return metrics"""
        logger.info("Evaluating model...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        logger.info(f"Model Metrics: {metrics}")
        
        return metrics
    
    def log_to_mlflow(self, params: dict, metrics: dict, model_path: str):
        """Log parameters, metrics, and model to MLflow"""
        logger.info("Logging to MLflow...")
        
        with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(
                self.model,
                "model",
                registered_model_name=self.config['mlflow']['model_name']
            )
            
            # Log scaler
            mlflow.log_artifact(model_path + "/scaler.pkl")
            
            # Log config
            mlflow.log_artifact("config.yaml")
            
            logger.info(f"MLflow run completed. Run ID: {mlflow.active_run().info.run_id}")
    
    def save_model(self, model_dir: str = "models"):
        """Save model and artifacts locally"""
        logger.info(f"Saving model to {model_dir}")
        
        # Create directory
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "model.pkl")
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        
        logger.info("Model saved successfully")
        
        return model_dir
    
    def load_model(self, model_dir: str = "models"):
        """Load saved model"""
        logger.info(f"Loading model from {model_dir}")
        
        model_path = os.path.join(model_dir, "model.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        logger.info("Model loaded successfully")
    
    def predict(self, X):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")
        
        # Preprocess
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def run_pipeline(self, data_path: str):
        """Execute complete MLOps pipeline"""
        logger.info("=" * 80)
        logger.info("Starting MLOps Pipeline")
        logger.info("=" * 80)
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data(data_path)
        
        # Preprocess
        X_train_scaled, X_test_scaled = self.preprocess_data(X_train, X_test)
        
        # Train
        best_params, best_score = self.train_model(X_train_scaled, y_train)
        
        # Evaluate
        metrics = self.evaluate_model(X_test_scaled, y_test)
        
        # Save model
        model_dir = self.save_model()
        
        # Log to MLflow
        self.log_to_mlflow(best_params, metrics, model_dir)
        
        logger.info("=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 80)
        
        return metrics


def main():
    """Main execution function"""
    # Initialize pipeline
    pipeline = MLOpsPipeline(config_path="config.yaml")
    
    # Run pipeline
    data_path = "data/train.csv"
    
    if os.path.exists(data_path):
        metrics = pipeline.run_pipeline(data_path)
        print(f"\nFinal Model Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    else:
        logger.warning(f"Data file not found: {data_path}")
        logger.info("Please add your training data to the data/ directory")


if __name__ == "__main__":
    main()
