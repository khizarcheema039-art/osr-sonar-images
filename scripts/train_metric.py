"""
Script: train_metric.py
Trains the EmbeddingNet with Triplet Margin Loss and online hard negative mining.
Saves model weights for Deep Metric Learning OSR inference.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from configs.config import (
    METRIC_EPOCHS,
    METRIC_LR,
    EMBEDDING_DIM,
    TRIPLET_MARGIN,
    DEVICE,
    MODELS_DIR,
)
from src.dataset import create_splits, get_dataloaders
from src.models import EmbeddingNet
from src.training import MetricTrainer


def main():
    print(f"Using device: {DEVICE}")
    print("Preparing dataset splits for Deep Metric Learning...")
    splits = create_splits()
    loaders = get_dataloaders(splits)

    # Build metric embedding network
    model = EmbeddingNet(embedding_dim=EMBEDDING_DIM, pretrained=True)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=METRIC_LR,
        weight_decay=1e-4,
    )
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    save_path = MODELS_DIR / "metric_embedding_model.pth"
    trainer = MetricTrainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        optimizer=optimizer,
        scheduler=scheduler,
        margin=TRIPLET_MARGIN,
        device=DEVICE,
        patience=5,
        save_path=save_path,
    )

    history = trainer.fit(num_epochs=METRIC_EPOCHS)
    print("Deep Metric Learning training pipeline complete!")


if __name__ == "__main__":
    main()
