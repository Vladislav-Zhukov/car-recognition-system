from fastapi import FastAPI

from app.api.routes.auth import router as auth_router
from app.api.routes.prediction import router as prediction_router
from app.api.routes.users import router as users_router

app = FastAPI(
    title="Car Recognition API",
    description="API for car make/model recognition using EfficientNet-B0.",
    version="1.0.0",
)

app.include_router(auth_router)
app.include_router(users_router)
app.include_router(prediction_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}