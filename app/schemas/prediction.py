from datetime import datetime

from pydantic import BaseModel


class PredictionItem(BaseModel):
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
    filename: str
    predictions: list[PredictionItem]
    model_name: str
    model_version: str


class PredictionHistoryItem(BaseModel):
    id: int
    filename: str
    top_prediction: str | None = None
    confidence: float | None = None
    created_at: datetime
    task_id: str | None = None
    status: str
    model_name: str
    model_version: str

    class Config:
        from_attributes = True