import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

from ml.core.config import (
    BATCH_SIZE,
    DEVICE,
    FAIR_CUSTOM_CNN_MODEL_PATH,
    FAIR_EFFICIENTNET_MODEL_PATH,
    IMAGE_SIZE,
    VAL_DIR,
)
from ml.models.custom_cnn import create_custom_cnn
from ml.models.efficientnet import create_model


def get_loader_for_efficientnet():
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
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return dataset, loader


def get_loader_for_custom_cnn():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return dataset, loader


def evaluate_model(model, loader, classes_count):
    model.eval()

    total = 0
    top1_correct = 0
    top5_correct = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            _, top1_preds = torch.max(probabilities, dim=1)

            top5_preds = torch.topk(
                probabilities,
                k=min(5, classes_count),
                dim=1,
            ).indices

            top1_correct += (top1_preds == labels).sum().item()

            top5_correct += sum(
                labels[i].item() in top5_preds[i].tolist()
                for i in range(labels.size(0))
            )

            total += labels.size(0)

    return top1_correct / total, top5_correct / total


def load_efficientnet(num_classes):
    checkpoint = torch.load(FAIR_EFFICIENTNET_MODEL_PATH, map_location=DEVICE)

    model = create_model(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)

    return model, checkpoint


def load_custom_cnn(num_classes):
    checkpoint = torch.load(FAIR_CUSTOM_CNN_MODEL_PATH, map_location=DEVICE)

    model = create_custom_cnn(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)

    return model, checkpoint


def main():
    eff_dataset, eff_loader = get_loader_for_efficientnet()
    cnn_dataset, cnn_loader = get_loader_for_custom_cnn()

    efficientnet, efficientnet_checkpoint = load_efficientnet(
        num_classes=len(eff_dataset.classes)
    )

    custom_cnn, custom_cnn_checkpoint = load_custom_cnn(
        num_classes=len(cnn_dataset.classes)
    )

    eff_top1, eff_top5 = evaluate_model(
        efficientnet,
        eff_loader,
        len(eff_dataset.classes),
    )

    cnn_top1, cnn_top5 = evaluate_model(
        custom_cnn,
        cnn_loader,
        len(cnn_dataset.classes),
    )

    print("\nFair Model Comparison")
    print("-" * 80)
    print(f"{'Model':<20} {'Epochs':<10} {'Top-1 Accuracy':<20} {'Top-5 Accuracy'}")
    print("-" * 80)

    print(
        f"{'EfficientNet-B0':<20} "
        f"{efficientnet_checkpoint.get('epochs', 'N/A'):<10} "
        f"{eff_top1 * 100:<20.2f} "
        f"{eff_top5 * 100:.2f}"
    )

    print(
        f"{'CustomCNN':<20} "
        f"{custom_cnn_checkpoint.get('epochs', 'N/A'):<10} "
        f"{cnn_top1 * 100:<20.2f} "
        f"{cnn_top5 * 100:.2f}"
    )


if __name__ == "__main__":
    main()