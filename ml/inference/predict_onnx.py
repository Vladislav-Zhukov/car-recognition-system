import sys

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision import models, transforms

from ml.core.config import FAIR_EFFICIENTNET_MODEL_PATH, IMAGE_SIZE, MODEL_DIR

ONNX_MODEL_PATH = MODEL_DIR / "fair_efficientnet_100.onnx"


def load_classes():
    checkpoint = torch.load(FAIR_EFFICIENTNET_MODEL_PATH, map_location="cpu")
    return checkpoint["classes"]


def build_transform():
    weights = models.EfficientNet_B0_Weights.DEFAULT

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=weights.transforms().mean,
            std=weights.transforms().std,
        ),
    ])


def softmax(x: np.ndarray):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def predict_onnx(image_path: str, top_k: int = 5):
    classes = load_classes()
    transform = build_transform()

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).numpy().astype(np.float32)

    session = ort.InferenceSession(
        str(ONNX_MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    outputs = session.run(
        None,
        {"input": image_tensor},
    )

    probabilities = softmax(outputs[0])

    top_indices = probabilities[0].argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:
        results.append({
            "class_name": classes[index],
            "confidence": round(float(probabilities[0][index] * 100), 2),
        })

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ml.inference.predict_onnx path_to_image")
        sys.exit(1)

    predictions = predict_onnx(sys.argv[1])

    print("\nONNX TOP-5 predictions:")

    for prediction in predictions:
        print(f"{prediction['class_name']}: {prediction['confidence']:.2f}%")