from app.core.celery_app import celery_app
from ml.inference.predict import predict_image


@celery_app.task(name="predict_car_task")
def predict_car_task(image_path: str):
    predictions = predict_image(image_path)

    return {
        "image_path": image_path,
        "predictions": predictions,
    }