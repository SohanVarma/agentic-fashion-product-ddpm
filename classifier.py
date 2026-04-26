import torch.nn as nn
import torch.nn.functional as F


class FashionClassifier(nn.Module):
    def __init__(self, num_classes: int, img_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(img_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        h = self.features(x)
        h = h.flatten(1)
        return self.classifier(h)
