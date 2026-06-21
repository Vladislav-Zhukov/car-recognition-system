# Car Recognition System

AI-powered car recognition platform built with FastAPI, PyTorch, PostgreSQL, Redis, MinIO, Kubernetes, and multiple background processing systems.

## Features

* JWT Authentication
* Car image upload
* Car recognition using Deep Learning models
* Synchronous predictions
* Asynchronous predictions
* Prediction history
* Object storage with MinIO
* Background processing with:

  * Celery
  * ARQ
  * TaskIQ
* Monitoring with:

  * Prometheus
  * Grafana
* Kubernetes deployment
* Horizontal Pod Autoscaler (HPA)
* Metrics Server
* Automated testing
* GitHub Actions CI

---

## Technology Stack

### Backend

* Python 3.12+
* FastAPI
* SQLAlchemy Async
* Alembic
* PostgreSQL
* Redis
* JWT Authentication

### Machine Learning

* PyTorch
* Torchvision
* ONNX

### Storage

* MinIO (S3-compatible object storage)

### Background Processing

* Celery
* ARQ
* TaskIQ

### Monitoring

* Prometheus
* Grafana
* FastAPI Metrics
* Kubernetes Metrics Server

### Infrastructure

* Docker
* Docker Compose
* Kubernetes
* Horizontal Pod Autoscaler (HPA)
* GitHub Actions

### Testing

* pytest
* pytest-asyncio
* httpx.AsyncClient

---

## Architecture

```text
User
 │
 ▼
FastAPI API
 │
 ├── PostgreSQL
 ├── Redis
 ├── MinIO
 │
 ├── Celery
 ├── ARQ
 └── TaskIQ
 │
 ▼
PyTorch Models
 │
 ▼
Prediction Results
```

---

## API Features

### Authentication

* Register
* Login
* JWT Access Token
* Refresh Token

### Prediction

#### Synchronous Prediction

```http
POST /predict
```

Upload image and receive prediction immediately.

#### Prediction History

```http
GET /predictions
GET /predictions/{prediction_id}
```

#### Celery Prediction

```http
POST /predict/async
GET /tasks/{task_id}
```

Submit image for background processing.

#### ARQ Prediction

```http
POST /arq/predict
GET /arq/tasks/{job_id}
```

Submit image to ARQ queue.

#### TaskIQ Prediction

```http
POST /taskiq/predict
GET /taskiq/tasks/{task_id}
```

Submit image to TaskIQ queue.

#### Monitoring

```http
GET /health
GET /metrics
```

---

## Local Development

### Clone Repository

```bash
git clone <repository-url>
cd car-recognition-system
```

### Environment

Create `.env`:

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/car_recognition

REDIS_URL=redis://redis:6379/0

S3_ENDPOINT_URL=http://minio:9000
S3_ACCESS_KEY=minio
S3_SECRET_KEY=minio12345
S3_BUCKET_NAME=car-images

SECRET_KEY=secretkeyforcarrecognition1278
POSTGRES_PASSWORD=postgres
```

### Run with Docker Compose

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

### Apply Migrations

```bash
docker compose -f docker-compose.prod.yml exec api alembic upgrade head
```

### Open API

Swagger:

```text
http://localhost:8000/docs
```

---

## Kubernetes Deployment

The project includes Kubernetes manifests located in the `k8s/` directory.

### Implemented Kubernetes Resources

* Namespace
* Secret
* ConfigMap
* Deployments
* Services
* PersistentVolumeClaims
* HorizontalPodAutoscaler
* Metrics Server

### Kubernetes Components

* FastAPI API
* PostgreSQL
* Redis
* MinIO
* Celery Worker
* ARQ Worker
* TaskIQ Worker
* Prometheus
* Grafana

### Deploy

```bash
kubectl apply -f k8s/
```

### Run Migrations

```bash
kubectl exec -it deployment/api -n car-recognition -- alembic upgrade head
```

### Access API

```bash
kubectl port-forward svc/api 8000:8000 -n car-recognition
```

Swagger:

```text
http://localhost:8000/docs
```

### Useful Commands

```bash
kubectl get pods -n car-recognition
kubectl get svc -n car-recognition
kubectl get hpa -n car-recognition

kubectl top nodes
kubectl top pods -n car-recognition
```

---

## Monitoring

### Prometheus

Metrics endpoint:

```text
/metrics
```

### Grafana

Dashboard visualization and monitoring.

### Metrics Server

Provides CPU and memory metrics for Kubernetes resources.

### Horizontal Pod Autoscaler

Configured for automatic API scaling based on CPU utilization.

---

## Storage

Images are stored in MinIO.

Bucket:

```text
car-images
```

Internal URL:

```text
http://minio:9000
```

---

## Background Processing

The project demonstrates three different Python background processing approaches.

### Celery

Classic distributed task queue with Redis broker.

### ARQ

Async Redis Queue for asyncio-native background processing.

### TaskIQ

Modern async task processing framework.

---

## Machine Learning

Supported model formats:

* PyTorch (.pt)
* ONNX (.onnx)

Models are stored in:

```text
trained_models/
```

Example models:

* fair_custom_cnn_100.pt
* fair_efficientnet_100.pt

---

## Testing

Automated API tests implemented with:

* pytest
* pytest-asyncio
* httpx.AsyncClient

Covered functionality:

* User registration
* Duplicate user validation
* User login
* Invalid credentials handling
* JWT refresh tokens
* Health endpoint
* Metrics endpoint
* Prediction validation
* Prediction history
* Prediction lookup by ID
* Celery task status
* TaskIQ task status

Current test suite:

```text
17 passed
```

Run tests:

```bash
pytest -v
```

---

## CI/CD

GitHub Actions workflow performs:

* Dependency installation
* Python syntax validation
* FastAPI import validation
* Automated test execution
* Docker image build
* Kubernetes YAML validation

---

## Future Improvements

* MLflow experiment tracking
* Model Registry
* KServe deployment
* Canary Deployments
* Model Monitoring

---

## Author

Vladislav Zhukov

Portfolio project for Python Backend / AI Backend Engineer positions.
