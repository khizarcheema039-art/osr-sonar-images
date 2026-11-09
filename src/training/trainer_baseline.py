"""
Training and evaluation loop for the ResNet-50 baseline closed-set classifier.
"""

import time
import copy
from pathlib import Path
import torch
import torch.nn as nn
from src.training.early_stopping import EarlyStopping
from configs.config import MODELS_DIR, DEVICE


class BaselineTrainer:
    """
    Handles training, validation, learning rate scheduling, early stopping,
    and checkpoint saving for ResNet-50.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=None,
        device=DEVICE,
        patience=5,
        save_path=MODELS_DIR / "resnet50_baseline.pth",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_path = Path(save_path)
        self.early_stopping = EarlyStopping(patience=patience, mode="min")

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def evaluate(self, loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels, _ in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def fit(self, num_epochs=25):
        print(f"Training Baseline ResNet-50 for up to {num_epochs} epochs...")
        print("=" * 65)
        print(f"{'Epoch':>6} | {'T-Loss':>8} | {'T-Acc':>8} | {'V-Loss':>8} | {'V-Acc':>8} | {'Time':>6}")
        print("-" * 65)

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            t_loss, t_acc = self.train_epoch()
            v_loss, v_acc = self.evaluate(self.val_loader)

            if self.scheduler:
                self.scheduler.step()

            self.history["train_loss"].append(t_loss)
            self.history["train_acc"].append(t_acc)
            self.history["val_loss"].append(v_loss)
            self.history["val_acc"].append(v_acc)

            duration = time.time() - t0
            improved = self.early_stopping(v_loss, self.model)
            status = "✅ BEST" if improved else f"patience: {self.early_stopping.counter}/{self.early_stopping.patience}"

            print(
                f"{epoch:>6} | {t_loss:>8.4f} | {t_acc * 100:>7.2f}% | "
                f"{v_loss:>8.4f} | {v_acc * 100:>7.2f}% | {duration:>5.1f}s {status}"
            )

            if self.early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        total_time = time.time() - start_time
        print("=" * 65)
        print(f"Training completed in {total_time / 60:.2f} minutes.")

        # Load best model weights
        if self.early_stopping.best_model_weights is not None:
            self.model.load_state_dict(self.early_stopping.best_model_weights)
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.early_stopping.best_model_weights, self.save_path)
            print(f"Best model saved to {self.save_path}")

        return self.history
