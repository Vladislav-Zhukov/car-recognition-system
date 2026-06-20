from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from taskiq import AsyncTaskiqTask

from app.taskiq_tasks import predict_car_taskiq
from ml.core.config import UPLOAD_DIR

router = APIRouter(prefix="/taskiq", tags=["TaskIQ"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

taskiq_tasks_storage: dict[str, AsyncTaskiqTask] = {}


@router.post("/predict")
async def taskiq_predict(file: UploadFile = File(...)):
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

    task = await predict_car_taskiq.kiq(str(file_path))

    taskiq_tasks_storage[task.task_id] = task

    return {
        "task_id": task.task_id,
        "status": "queued",
    }


@router.get("/tasks/{task_id}")
async def get_taskiq_result(task_id: str):
    task = taskiq_tasks_storage.get(task_id)

    if task is None:
        return {
            "task_id": task_id,
            "status": "unknown",
            "detail": "Task was not found in local API memory. Restarting API clears this storage.",
        }

    if not await task.is_ready():
        return {
            "task_id": task_id,
            "status": "pending",
        }

    result = await task.wait_result()

    return {
        "task_id": task_id,
        "status": "completed",
        "result": result.return_value,
    }