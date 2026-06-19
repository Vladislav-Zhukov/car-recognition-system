from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from ml.core.config import BATCH_SIZE, DEVICE, IMAGE_SIZE, MODEL_PATH, REPORTS_DIR, VAL_DIR
from ml.models.efficientnet import create_model

CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"


def get_loader():
    weights = models.EfficientNet_B0_Weights.DEFAULT

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=weights.transforms().mean,
            std=weights.transforms().std,
        ),
    ])

    dataset = datasets.ImageFolder(VAL_DIR, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return dataset, loader


def load_trained_model(num_classes: int):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model = create_model(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    return model


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset, loader = get_loader()
    model = load_trained_model(num_classes=len(dataset.classes))

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Building confusion matrix"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(14, 14))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=dataset.classes,
    )

    display.plot(
        ax=ax,
        xticks_rotation=90,
        cmap="Blues",
        colorbar=True,
    )

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=200)

    print(f"\nSaved confusion matrix to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
