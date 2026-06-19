import time
from pathlib import Path

import onnxruntime as ort
import torch
from PIL import Image
from torchvision import models, transforms

from ml.core.config import (
    DEVICE,
    FAIR_EFFICIENTNET_MODEL_PATH,
    IMAGE_SIZE,
    MODEL_DIR,
)
from ml.models.efficientnet import create_model

ONNX_MODEL_PATH = MODEL_DIR / "fair_efficientnet_100.onnx"
TEST_IMAGE_PATH = Path("test.jpg")
ITERATIONS = 50


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


def load_image():
    transform = build_transform()
    image = Image.open(TEST_IMAGE_PATH).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    return tensor


def load_pytorch_model():
    checkpoint = torch.load(FAIR_EFFICIENTNET_MODEL_PATH, map_location=DEVICE)

    model = create_model(num_classes=len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    return model


def benchmark_pytorch():
    model = load_pytorch_model()
    image = load_image().to(DEVICE)

    with torch.no_grad():
        for _ in range(5):
            _ = model(image)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(ITERATIONS):
            _ = model(image)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

    avg_time = (end - start) / ITERATIONS

    return avg_time


def benchmark_onnx():
    session = ort.InferenceSession(
        str(ONNX_MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    image = load_image().numpy()

    for _ in range(5):
        _ = session.run(None, {"input": image})

    start = time.perf_counter()

    for _ in range(ITERATIONS):
        _ = session.run(None, {"input": image})

    end = time.perf_counter()

    avg_time = (end - start) / ITERATIONS

    return avg_time


def main():
    if not TEST_IMAGE_PATH.exists():
        raise FileNotFoundError("Put test.jpg in the project root first.")

    pytorch_time = benchmark_pytorch()
    onnx_time = benchmark_onnx()

    print("\nInference Benchmark")
    print("-" * 50)
    print(f"Device for PyTorch: {DEVICE}")
    print(f"Iterations: {ITERATIONS}")
    print("-" * 50)
    print(f"PyTorch avg time: {pytorch_time * 1000:.2f} ms")
    print(f"ONNX CPU avg time: {onnx_time * 1000:.2f} ms")


if __name__ == "__main__":
    main()