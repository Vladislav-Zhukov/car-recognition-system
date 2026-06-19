from pathlib import Path

from ml.inference.predict import predict_image


class CarPredictionService:
    def predict(self, image_path: Path):
        return predict_image(str(image_path))


car_prediction_service = CarPredictionService()
