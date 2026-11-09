"""
Deep Metric Learning (DML) Embedding Network.
Maps ResNet-50 GAP representations through a projection head to an L2-normalized hypersphere.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from configs.config import EMBEDDING_DIM


class EmbeddingNet(nn.Module):
    """
    ResNet-50 backbone + Multi-Layer Perceptron (MLP) projection head.
    Outputs L2-normalized embedding vectors optimized for Triplet Loss metric learning.
    """

    def __init__(self, embedding_dim=EMBEDDING_DIM, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Backbone up to GAP layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze early layers for stability
        for name, param in self.backbone.named_parameters():
            if "layer1" in name or "conv1" in name or "bn1" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        in_features = 2048  # GAP output dimension

        # Projection head: 2048 -> 512 -> embedding_dim
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x):
        """
        Forward pass yielding unit-normalized embeddings.
        Shape: (B, embedding_dim)
        """
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        embeddings = self.head(features)
        return F.normalize(embeddings, p=2, dim=1)
