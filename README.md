# MLOps Assignment 2: Cats vs Dogs Classification Pipeline

Complete end-to-end MLOps pipeline for binary image classification with CI/CD automation.

## Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Prepare dataset (if you have PetImages folder)
python3 scripts/prepare_dataset.py

# Or create dummy data for testing
python3 scripts/create_dummy_data.py --num-images 100
```

### 2. Train Model
```bash
# Train with MLflow tracking
python3 src/train.py --num_epochs 20 --batch_size 32

# View experiments
mlflow ui  # Open http://localhost:5000
```

### 3. Run API Server
```bash
# Start the FastAPI server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Test endpoints
curl http://localhost:8000/health
curl -X POST -F "file=@path/to/image.jpg" http://localhost:8000/predict
```

### 4. Run Tests
```bash
pytest tests/ -v --cov=src --cov=api
```

### 5. Docker Deployment
```bash
# Build image
docker build -t cats-dogs-classifier:latest .

# Run container
docker run -p 8000:8000 cats-dogs-classifier:latest

# Or use Docker Compose
docker-compose up
```

### 6. Kubernetes Deployment
```bash
# Start local cluster
minikube start

# Deploy application
kubectl apply -f deployment/k8s/deployment.yaml

# Port forward to access
kubectl port-forward service/cats-dogs-classifier-service 8000:80

# Run smoke tests
python3 deployment/smoke_test.py
```

## Project Structure

```
.
├── api/                        # FastAPI REST API
│   ├── __init__.py
│   └── main.py                 # API endpoints and monitoring
├── src/                        # ML pipeline code
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data loading and preprocessing
│   ├── model.py                # CNN model architecture
│   ├── train.py                # Training with MLflow tracking
│   └── inference.py            # Model inference utilities
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_preprocessing.py   # Data preprocessing tests
│   └── test_inference.py       # Model inference tests
├── deployment/                 # Deployment configurations
│   ├── k8s/
│   │   └── deployment.yaml     # Kubernetes manifests
│   ├── prometheus.yml          # Monitoring config
│   └── smoke_test.py           # Post-deployment validation
├── scripts/                    # Utility scripts
│   ├── prepare_dataset.py      # Dataset preparation
│   ├── create_dummy_data.py    # Test data generator
│   └── download_dataset.sh     # Kaggle download helper
├── .github/workflows/          # CI/CD pipeline
│   └── ci-cd.yml               # GitHub Actions workflow
├── data/                       # Dataset (gitignored)
├── models/                     # Trained models (gitignored)
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Multi-container setup
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check with model status |
| `/info` | GET | Model metadata |
| `/predict` | POST | Image classification (accepts image file) |
| `/metrics` | GET | Prometheus metrics |

## Dataset

- **Source:** Cats and Dogs from Kaggle or Microsoft PetImages
- **Size:** ~25,000 images (12,500 cats + 12,500 dogs)
- **Split:** 80% train / 10% validation / 10% test
- **Preprocessing:** Resize to 224x224, normalize, augmentation

## Model

- **Architecture:** Custom CNN with 4 convolutional blocks
- **Parameters:** ~26M trainable parameters
- **Features:** Batch normalization, dropout, max pooling
- **Expected Accuracy:** ~90% on test set (after 20 epochs)

## Monitoring

- **MLflow:** Experiment tracking, parameter/metric logging
- **Prometheus:** Request count, latency, prediction metrics
- **Logging:** Structured request/response logs

## CI/CD Pipeline

The GitHub Actions workflow automatically:
1. Runs unit tests on every push
2. Builds Docker image
3. Pushes to container registry
4. Deploys to Kubernetes (on main branch)
5. Runs smoke tests

## Development

### Run Locally
```bash
# Activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start development server
uvicorn api.main:app --reload
```

### Run Tests
```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov=api --cov-report=term

# Specific test file
pytest tests/test_preprocessing.py -v
```

### Code Quality
```bash
# Format code
black src/ api/ tests/

# Lint
pylint src/ api/

# Type checking
mypy src/ api/
```

## Troubleshooting

### Dataset Issues
- Ensure dataset is in `data/raw/cats/` and `data/raw/dogs/`
- Run `python3 scripts/prepare_dataset.py` to clean corrupted images

### Model Not Found
- Train model first: `python3 src/train.py`
- Model saved to: `models/best_model.pth`

### Port Already in Use
```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9
```

### Docker Build Fails
- Ensure model file exists in `models/` directory
- Check Docker daemon is running

## Assignment Requirements

- ✅ **M1:** Model development with Git/DVC versioning and MLflow tracking
- ✅ **M2:** FastAPI service with containerization
- ✅ **M3:** CI pipeline with automated testing and image building
- ✅ **M4:** CD pipeline with Kubernetes deployment and smoke tests
- ✅ **M5:** Monitoring with Prometheus and structured logging

## License

MIT

## Authors

MLOps Assignment 2 - BITS Pilani (S1-25_AIMLCZG523)
