from pathlib import Path

from src.predict import predict_image


class CarPredictorService:
    def predict(self, image_path: Path):
        return predict_image(str(image_path))


car_predictor_service = CarPredictorService()