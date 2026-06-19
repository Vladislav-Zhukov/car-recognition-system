import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File

from app.schemas.prediction import PredictionResponse
from app.services.predictor import car_predictor_service

router = APIRouter(prefix="/predict", tags=["Prediction"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("", response_model=PredictionResponse)
async def predict_car(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    predictions = car_predictor_service.predict(file_path)

    return {
        "filename": file.filename,
        "predictions": predictions,
    }