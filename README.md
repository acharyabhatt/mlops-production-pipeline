# 🚀 MLOps Pipeline with MLflow and DVC

A production-ready MLOps pipeline featuring experiment tracking, model versioning, monitoring, and automated deployment.

## 🎯 Features

- **Experiment Tracking**: MLflow for tracking parameters, metrics, and models
- **Data Versioning**: DVC for dataset and model versioning
- **Model Training**: Automated hyperparameter tuning with scikit-learn
- **Model Serving**: FastAPI REST API with Prometheus metrics
- **Monitoring**: Prediction logging and model drift detection
- **CI/CD Ready**: GitHub Actions workflows included
- **Containerization**: Docker support for easy deployment

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Data      │────▶│   Training   │────▶│   MLflow    │
│   (DVC)     │     │   Pipeline   │     │  Tracking   │
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │    Model     │
                    │   Registry   │
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐     ┌─────────────┐
                    │   FastAPI    │────▶│ Prometheus  │
                    │   Service    │     │ Monitoring  │
                    └──────────────┘     └─────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.9+
- Docker (optional)
- AWS/GCP credentials (for DVC remote storage)

### Setup

```bash
# Clone repository
git clone <your-repo-url>
cd mlops-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize DVC
dvc init

# Setup MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000
```

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
# Model parameters
model:
  name: "RandomForestClassifier"
  param_grid:
    n_estimators: [100, 200, 300]
    max_depth: [10, 20, 30]

# MLflow settings
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "ml-pipeline-experiment"
```

## 💻 Usage

### 1. Prepare Data

```bash
# Add data to DVC
dvc add data/train.csv
git add data/train.csv.dvc data/.gitignore
git commit -m "Add training data"

# Push to remote storage
dvc push
```

### 2. Train Model

```bash
python train.py
```

This will:
- Load and preprocess data
- Train model with hyperparameter tuning
- Log experiments to MLflow
- Save model artifacts

### 3. Serve Model

```bash
# Start API server
python serve.py

# Or using uvicorn
uvicorn serve:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Make Predictions

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [1.0, 2.0, 3.0, 4.0],
    "request_id": "test-001"
  }'

# Batch prediction
curl -X POST "http://localhost:8000/batch-predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
  }'
```

## 📊 MLflow UI

Access MLflow UI at `http://localhost:5000`

Features:
- Compare experiments
- View metrics and parameters
- Download models
- Register models to production

## 🐳 Docker Deployment

```bash
# Build image
docker build -t ml-pipeline:latest .

# Run container
docker run -p 8000:8000 ml-pipeline:latest

# Or use docker-compose
docker-compose up
```

## 📈 Monitoring

### Prometheus Metrics

Access metrics at `http://localhost:8000/metrics`

Available metrics:
- Request count
- Request duration
- Prediction distribution
- Model performance

### Prediction Logging

Predictions are logged to `logs/predictions.jsonl` for:
- Model drift detection
- A/B testing
- Performance analysis

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=. tests/

# Lint code
flake8 .

# Format code
black .
```

## 📁 Project Structure

```
mlops-pipeline/
├── train.py              # Training pipeline
├── serve.py              # FastAPI serving
├── config.yaml           # Configuration
├── requirements.txt      # Dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose
├── .dvc/                # DVC configuration
├── data/                # Data directory (DVC tracked)
│   └── train.csv
├── models/              # Model artifacts
│   ├── model.pkl
│   └── scaler.pkl
├── logs/                # Prediction logs
├── tests/               # Unit tests
├── .github/
│   └── workflows/       # CI/CD workflows
└── README.md
```

## 🔄 CI/CD Pipeline

GitHub Actions workflows included:

1. **Test**: Run tests on PR
2. **Train**: Retrain model on data changes
3. **Deploy**: Deploy to production on merge

## 📝 Best Practices

### Model Training
- Always version your data with DVC
- Log all experiments to MLflow
- Use cross-validation for model selection
- Save preprocessing artifacts

### Model Serving
- Use FastAPI for async performance
- Implement health checks
- Log predictions for monitoring
- Enable caching for frequently requested predictions

### Monitoring
- Track prediction distribution
- Monitor model drift
- Set up alerts for performance degradation
- Review prediction logs regularly

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

MIT License

## 🙏 Acknowledgments

- MLflow for experiment tracking
- DVC for data versioning
- FastAPI for serving
- Prometheus for monitoring

## 📧 Contact

For questions or suggestions, please open an issue.
