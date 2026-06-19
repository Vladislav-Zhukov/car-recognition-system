# 🚗 Car Recognition System

AI Backend project for car make and model recognition from images using Deep Learning and FastAPI.

## 📌 About Project

Car Recognition System is a machine learning application that identifies a car make and model from an uploaded image.

The project combines:

* Deep Learning (PyTorch)
* Computer Vision
* REST API (FastAPI)
* PostgreSQL
* Docker
* Model Evaluation and Analytics

The goal of this project is to understand the complete ML lifecycle:

1. Dataset preparation
2. Model training
3. Evaluation
4. Deployment as API
5. Prediction history storage
6. Future model improvements

---

# 🛠 Tech Stack

### Backend

* Python 3.12
* FastAPI
* SQLAlchemy Async
* Alembic
* PostgreSQL
* Docker
* Docker Compose

### Machine Learning

* PyTorch
* Torchvision
* EfficientNet-B0
* Transfer Learning

### Data Processing

* PIL (Pillow)
* NumPy
* Pandas

### Visualization

* Matplotlib
* Scikit-learn

---

# 📂 Project Structure

```text
car-recognition-system/

├── app/
│   ├── api/
│   ├── db/
│   ├── schemas/
│   ├── services/
│   └── main.py
│
├── ml/
│   ├── core/
│   ├── models/
│   ├── training/
│   ├── inference/
│   ├── evaluation/
│   └── data/
│
├── dataset/
│
├── trained_models/
│
├── reports/
│
├── uploads/
│
├── alembic/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# 🧠 Model

Current implementation uses:

**EfficientNet-B0**

with Transfer Learning.

The final classification layer was replaced and trained on a subset of Stanford Cars Dataset.

---

# 📊 Dataset

Dataset:

Stanford Cars Dataset

Contains images of different car makes and models.

For faster experimentation, a subset of 10 classes was used:

* Audi
* BMW
* Chevrolet
* Dodge
* Ford
* Mercedes-Benz
* Porsche
* Tesla

---

# 📈 Model Results

Current evaluation results:

```text
Top-1 Accuracy: 72.26%
Top-5 Accuracy: 97.81%
```

Meaning:

* Correct prediction is first choice in 72% of cases.
* Correct prediction is among top-5 predictions in almost 98% of cases.

---

# 🔍 Evaluation

Model evaluation includes:

* Top-1 Accuracy
* Top-5 Accuracy
* Confusion Matrix

Generate evaluation:

```bash
python -m ml.evaluation.evaluate
```

Generate confusion matrix:

```bash
python -m ml.evaluation.confusion_matrix
```

---

# 🚀 Training

Train model:

```bash
python -m ml.training.train
```

The best model is automatically saved to:

```text
trained_models/car_model.pt
```

---

# 🔎 Prediction

CLI prediction:

```bash
python -m ml.inference.predict test.jpg
```

Example output:

```text
TOP-5 predictions:

BMW M3 Coupe 2012: 84.12%
Ford Mustang Convertible 2007: 6.51%
Audi S4 Sedan 2012: 4.33%
...
```

---

# 🌐 REST API

Run FastAPI:

```bash
uvicorn app.main:app --reload
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

Available endpoints:

```http
GET /health

POST /predict

GET /predictions
```

Example response:

```json
{
  "filename": "car.jpg",
  "predictions": [
    {
      "class_name": "BMW M3 Coupe 2012",
      "confidence": 84.12
    }
  ]
}
```

---

# 🐳 Docker

Start PostgreSQL:

```bash
docker compose up -d db
```

Start full application:

```bash
docker compose up --build
```

---

# 💾 Database

Prediction history is stored in PostgreSQL.

Stored fields:

* filename
* predicted class
* confidence
* prediction date

---

# 📋 Future Improvements

Planned improvements:

* Custom CNN implementation
* EfficientNet vs Custom CNN comparison
* Larger dataset
* User authentication
* Frontend application
* Model versioning
* Cloud deployment


1. Custom CNN
2. Сравнение CNN vs EfficientNet
3. ONNX Export
4. Redis + Celery
5. GitHub Actions
6. VPS Deploy