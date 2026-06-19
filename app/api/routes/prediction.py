import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.api.rate_limit import check_rate_limit
from app.db.database import get_db
from app.db.models import Prediction, User
from app.core.celery_app import celery_app
from app.tasks import predict_car_task
from app.schemas.prediction import PredictionHistoryItem, PredictionResponse
from app.services.prediction_service import car_prediction_service
from ml.core.config import MODEL_NAME, MODEL_VERSION, UPLOAD_DIR
from celery.result import AsyncResult

router = APIRouter(tags=["Prediction"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@router.post("/predict", response_model=PredictionResponse)
async def predict_car(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await check_rate_limit(current_user.id)

    filename = file.filename or "uploaded_image.jpg"
    file_ext = Path(filename).suffix.lower()

    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use jpg, jpeg, png or webp.",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    predictions = car_prediction_service.predict(file_path)
    top_prediction = predictions[0]

    prediction_record = Prediction(
        filename=filename,
        top_prediction=top_prediction["class_name"],
        confidence=top_prediction["confidence"],
        user_id=current_user.id,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
    )

    db.add(prediction_record)
    await db.commit()

    return {
        "filename": filename,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "predictions": predictions,
    }


@router.get("/predictions", response_model=list[PredictionHistoryItem])
async def get_predictions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Prediction)
        .where(Prediction.user_id == current_user.id)
        .order_by(Prediction.created_at.desc())
    )

    return result.scalars().all()


@router.post("/predict/async")
async def predict_car_async(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    await check_rate_limit(current_user.id)

    filename = file.filename or "uploaded_image.jpg"
    file_ext = Path(filename).suffix.lower()

    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use jpg, jpeg, png or webp.",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    task = predict_car_task.delay(str(file_path))

    return {
        "task_id": task.id,
        "status": "queued",
    }


@router.get("/tasks/{task_id}")
async def get_task_result(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)

    if task_result.state == "PENDING":
        return {
            "task_id": task_id,
            "status": "pending",
        }

    if task_result.state == "STARTED":
        return {
            "task_id": task_id,
            "status": "started",
        }

    if task_result.state == "SUCCESS":
        return {
            "task_id": task_id,
            "status": "completed",
            "result": task_result.result,
        }

    if task_result.state == "FAILURE":
        return {
            "task_id": task_id,
            "status": "failed",
            "error": str(task_result.result),
        }

    return {
        "task_id": task_id,
        "status": task_result.state,
    }


@router.get("/predictions/{prediction_id}", response_model=PredictionHistoryItem)
async def get_prediction_by_id(
    prediction_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Prediction).where(
            Prediction.id == prediction_id,
            Prediction.user_id == current_user.id,
        )
    )

    prediction = result.scalar_one_or_none()

    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found",
        )

    return prediction