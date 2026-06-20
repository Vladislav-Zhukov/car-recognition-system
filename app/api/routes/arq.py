from pathlib import Path

from arq import create_pool
from arq.connections import RedisSettings
from arq.jobs import Job
from app.arq_worker import get_redis_settings
from fastapi import APIRouter, File, HTTPException, UploadFile

from ml.core.config import UPLOAD_DIR

router = APIRouter(prefix="/arq", tags=["ARQ"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@router.post("/predict")
async def arq_predict(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_image.jpg"
    file_ext = Path(filename).suffix.lower()

    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use jpg, jpeg, png or webp.",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / filename

    file_bytes = await file.read()

    with open(file_path, "wb") as buffer:
        buffer.write(file_bytes)

    redis = await create_pool(get_redis_settings())

    job = await redis.enqueue_job(
        "predict_car_arq",
        str(file_path),
    )

    return {
        "job_id": job.job_id,
        "status": "queued",
    }


@router.get("/tasks/{job_id}")
async def get_arq_task_result(job_id: str):
    redis = await create_pool(get_redis_settings())

    job = Job(
        job_id=job_id,
        redis=redis,
    )

    result = await job.result(timeout=0)

    if result is None:
        return {
            "job_id": job_id,
            "status": "pending",
        }

    return {
        "job_id": job_id,
        "status": "completed",
        "result": result,
    }