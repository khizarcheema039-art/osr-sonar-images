"""
Model architectures for Closed-Set and Open-Set Recognition.
"""

from src.models.resnet_baseline import ResNet50Baseline, build_resnet50
from src.models.embedding_net import EmbeddingNet

__all__ = ["ResNet50Baseline", "build_resnet50", "EmbeddingNet"]
