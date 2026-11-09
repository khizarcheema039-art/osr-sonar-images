"""
ResNet-50 Backbone and Closed-Set Classifier for Underwater Sonar Images.
Provides both classification logits and 2048-dimensional GAP feature representations.
"""

import torch
import torch.nn as nn
from torchvision import models
from configs.config import NUM_CLASSES


class ResNet50Baseline(nn.Module):
    """
    ResNet-50 model with transfer learning.
    Early layers (conv1, bn1, layer1) are frozen to preserve general low-level filters,
    while deeper residual blocks (layer2-4) and classification head are fine-tuned on sonar data.
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, dropout=0.3):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Freeze primitive feature layers
        for name, param in resnet.named_parameters():
            if "layer1" in name or "conv1" in name or "bn1" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Backbone up to Global Average Pooling (GAP)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        in_features = resnet.fc.in_features  # 2048

        # Classification Head
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def extract_features(self, x):
        """
        Extracts 2048-dimensional GAP latent representations.
        Output shape: (B, 2048)
        """
        feats = self.backbone(x)
        return torch.flatten(feats, 1)

    def forward(self, x):
        """
        Forward pass returning classification logits.
        """
        feats = self.extract_features(x)
        logits = self.fc(feats)
        return logits

    def forward_with_features(self, x):
        """
        Returns both features and logits simultaneously.
        """
        feats = self.extract_features(x)
        logits = self.fc(feats)
        return logits, feats


def build_resnet50(num_classes=NUM_CLASSES, pretrained=True):
    return ResNet50Baseline(num_classes=num_classes, pretrained=pretrained)
