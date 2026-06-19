import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from ml.core.config import (
    BATCH_SIZE,
    DEVICE,
    FAIR_CUSTOM_CNN_MODEL_PATH,
    FAIR_EPOCHS,
    IMAGE_SIZE,
    LEARNING_RATE,
    MODEL_DIR,
    TRAIN_DIR,
    VAL_DIR,
)
from ml.models.custom_cnn import create_custom_cnn


def get_loaders():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_dataset, val_dataset, train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training Fair CustomCNN"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation Fair CustomCNN"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def main():
    print(f"Using device: {DEVICE}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, train_loader, val_loader = get_loaders()

    model = create_custom_cnn(num_classes=len(train_dataset.classes))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(FAIR_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{FAIR_EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": train_dataset.classes,
                    "image_size": IMAGE_SIZE,
                    "best_val_acc": best_val_acc,
                    "model_name": "CustomCNN",
                    "model_version": "fair-100-epochs",
                    "epochs": FAIR_EPOCHS,
                },
                FAIR_CUSTOM_CNN_MODEL_PATH,
            )

            print(f"Best Fair CustomCNN saved: {FAIR_CUSTOM_CNN_MODEL_PATH}")

    print("\nFair CustomCNN training finished")
    print(f"Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()