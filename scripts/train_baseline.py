"""
Script: train_baseline.py
Trains the ResNet-50 closed-set classifier on known marine debris classes.
Extracts baseline classification report and saves checkpoint weights.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report

from configs.config import (
    NUM_CLASSES,
    BASELINE_EPOCHS,
    BASELINE_LR,
    BASELINE_PATIENCE,
    STEP_SIZE,
    GAMMA,
    DEVICE,
    MODELS_DIR,
    KNOWN_CLASSES,
    CLASS_TO_IDX,
)
from src.dataset import create_splits, get_dataloaders
from src.models import build_resnet50
from src.training import BaselineTrainer


def main():
    print(f"Using device: {DEVICE}")
    print("Preparing dataset splits (zero leakage)...")
    splits = create_splits()
    loaders = get_dataloaders(splits)

    print(f"Train samples: {len(splits['train'])}")
    print(f"Val samples:   {len(splits['val'])}")
    print(f"Test known:    {len(splits['test_known'])}")
    print(f"Test unknown:  {len(splits['test_unknown'])}")

    # Build model
    model = build_resnet50(num_classes=NUM_CLASSES, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=BASELINE_LR,
        weight_decay=1e-4,
    )
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    save_path = MODELS_DIR / "resnet50_baseline.pth"
    trainer = BaselineTrainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        patience=BASELINE_PATIENCE,
        save_path=save_path,
    )

    history = trainer.fit(num_epochs=BASELINE_EPOCHS)

    # Final evaluation on Known Test Set
    print("\nEvaluating Baseline on Known Test Set...")
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels, _ in loaders["test_known"]:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    target_names = [KNOWN_CLASSES[i] for i in range(NUM_CLASSES)]
    report = classification_report(all_targets, all_preds, target_names=target_names)
    print("\nClosed-Set Classification Report:")
    print("=" * 60)
    print(report)


if __name__ == "__main__":
    main()
