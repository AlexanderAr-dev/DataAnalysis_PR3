import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet


class SmallCNNAdaptive(nn.Module):
    def __init__(self, num_classes=6):
        super(SmallCNNAdaptive, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_resnet18(num_classes=6):
    """ResNet18 без предобученных весов для избежания проблем с загрузкой"""
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    print("Using ResNet18 with random initialization (no pretrained weights)")
    return model


def get_efficientnet_b0(num_classes=6):
    """EfficientNet-B0 без предобученных весов"""
    model = EfficientNet.from_name('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    print("Using EfficientNet-B0 with random initialization (no pretrained weights)")
    return model