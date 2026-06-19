from torch import nn
from torchvision import models


def create_model(num_classes: int, freeze_backbone: bool = True):
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model
