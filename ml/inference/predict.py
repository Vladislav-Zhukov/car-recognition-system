import sys

import torch
from PIL import Image
from torchvision import models, transforms

from ml.core.config import DEVICE, IMAGE_SIZE, MODEL_PATH
from ml.models.efficientnet import create_model


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    classes = checkpoint["classes"]

    model = create_model(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    return model, classes


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


def predict_image(image_path: str, top_k: int = 5):
    model, classes = load_model()
    transform = build_transform()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)

        top_probs, top_indices = torch.topk(
            probabilities,
            k=min(top_k, len(classes)),
        )

    results = []

    for prob, index in zip(top_probs[0], top_indices[0]):
        class_name = classes[index.item()]
        confidence = prob.item() * 100

        results.append({
            "class_name": class_name,
            "confidence": round(confidence, 2),
        })

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ml.inference.predict path_to_image")
        sys.exit(1)

    predictions = predict_image(sys.argv[1])

    print("\nTOP-5 predictions:")

    for prediction in predictions:
        print(f"{prediction['class_name']}: {prediction['confidence']:.2f}%")
