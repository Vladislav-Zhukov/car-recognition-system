import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from ml.core.config import BATCH_SIZE, DEVICE, IMAGE_SIZE, MODEL_DIR, VAL_DIR
from ml.models.custom_cnn import create_custom_cnn

CUSTOM_CNN_MODEL_PATH = MODEL_DIR / "custom_cnn.pt"


def get_test_loader():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(VAL_DIR, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return dataset, loader


def load_model(num_classes: int):
    checkpoint = torch.load(CUSTOM_CNN_MODEL_PATH, map_location=DEVICE)

    model = create_custom_cnn(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    return model


def evaluate():
    dataset, loader = get_test_loader()
    model = load_model(num_classes=len(dataset.classes))

    total = 0
    top1_correct = 0
    top5_correct = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating CustomCNN"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            _, top1_preds = torch.max(probabilities, dim=1)

            top5_preds = torch.topk(
                probabilities,
                k=min(5, len(dataset.classes)),
                dim=1,
            ).indices

            top1_correct += (top1_preds == labels).sum().item()

            top5_correct += sum(
                labels[i].item() in top5_preds[i].tolist()
                for i in range(labels.size(0))
            )

            total += labels.size(0)

    top1_accuracy = top1_correct / total
    top5_accuracy = top5_correct / total

    print("\nCustomCNN evaluation results:")
    print(f"Classes count: {len(dataset.classes)}")
    print(f"Images count: {total}")
    print(f"Top-1 Accuracy: {top1_accuracy:.4f} ({top1_accuracy * 100:.2f}%)")
    print(f"Top-5 Accuracy: {top5_accuracy:.4f} ({top5_accuracy * 100:.2f}%)")


if __name__ == "__main__":
    evaluate()