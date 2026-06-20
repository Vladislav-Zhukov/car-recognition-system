from fastapi import FastAPI

from app.api.routes.auth import router as auth_router
from app.api.routes.prediction import router as prediction_router
from app.api.routes.users import router as users_router
from app.api.routes.arq import router as arq_router
from app.core.metrics import PrometheusMiddleware, metrics_endpoint
from app.api.routes.taskiq import router as taskiq_router

app = FastAPI(
    title="Car Recognition API",
    description="AI Backend service for car recognition.",
    version="1.0.0",
)

app.add_middleware(PrometheusMiddleware)

app.include_router(auth_router)
app.include_router(users_router)
app.include_router(prediction_router)
app.include_router(arq_router)
app.include_router(taskiq_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return metrics_endpoint()