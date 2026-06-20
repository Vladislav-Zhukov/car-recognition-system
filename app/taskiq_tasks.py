from app.taskiq_broker import broker
from ml.inference.predict import predict_image


@broker.task
async def predict_car_taskiq(image_path: str):
    predictions = predict_image(image_path)

    return {
        "status": "completed",
        "image_path": image_path,
        "predictions": predictions,
    }