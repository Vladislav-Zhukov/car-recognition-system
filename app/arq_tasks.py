from ml.inference.predict import predict_image


async def predict_car_arq(ctx, image_path: str):
    predictions = predict_image(image_path)

    return {
        "status": "completed",
        "image_path": image_path,
        "predictions": predictions,
    }