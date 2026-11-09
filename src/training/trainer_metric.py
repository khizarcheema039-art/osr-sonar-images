"""
Deep Metric Learning Trainer utilizing Triplet Margin Loss and Online Hard Triplet Mining.
"""

import time
import copy
from pathlib import Path
import torch
from pytorch_metric_learning import losses, miners
from src.training.early_stopping import EarlyStopping
from configs.config import MODELS_DIR, DEVICE, TRIPLET_MARGIN


class MetricTrainer:
    """
    Trains EmbeddingNet using TripletMarginLoss with online hard negative mining.
    Forces known classes into hyper-compact clusters, reserving the outer embedding space
    for novel/unknown anomaly detection.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler=None,
        margin=TRIPLET_MARGIN,
        device=DEVICE,
        patience=5,
        save_path=MODELS_DIR / "metric_embedding_model.pth",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_path = Path(save_path)
        self.early_stopping = EarlyStopping(patience=patience, mode="min")

        self.loss_func = losses.TripletMarginLoss(margin=margin)
        self.miner = miners.MultiSimilarityMiner(epsilon=0.1)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "num_triplets": [],
        }

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_triplets = 0
        num_batches = 0

        for images, labels, _ in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            embeddings = self.model(images)

            # Mine informative triplets online
            hard_pairs = self.miner(embeddings, labels)
            loss = self.loss_func(embeddings, labels, hard_pairs)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_triplets += self.miner.num_triplets
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_triplets = total_triplets / max(num_batches, 1)
        return avg_loss, avg_triplets

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for images, labels, _ in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                embeddings = self.model(images)
                hard_pairs = self.miner(embeddings, labels)
                loss = self.loss_func(embeddings, labels, hard_pairs)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def fit(self, num_epochs=30):
        print(f"Training Deep Metric Learning Network for up to {num_epochs} epochs...")
        print("=" * 65)
        print(f"{'Epoch':>6} | {'T-Loss':>8} | {'V-Loss':>8} | {'Avg Triplets':>12} | {'Time':>6}")
        print("-" * 65)

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            t_loss, avg_triplets = self.train_epoch()
            v_loss = self.validate()

            if self.scheduler:
                self.scheduler.step()

            self.history["train_loss"].append(t_loss)
            self.history["val_loss"].append(v_loss)
            self.history["num_triplets"].append(avg_triplets)

            duration = time.time() - t0
            improved = self.early_stopping(v_loss, self.model)
            status = "✅ BEST" if improved else f"patience: {self.early_stopping.counter}/{self.early_stopping.patience}"

            print(
                f"{epoch:>6} | {t_loss:>8.4f} | {v_loss:>8.4f} | "
                f"{avg_triplets:>12.1f} | {duration:>5.1f}s {status}"
            )

            if self.early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        total_time = time.time() - start_time
        print("=" * 65)
        print(f"Metric training completed in {total_time / 60:.2f} minutes.")

        # Save best model
        if self.early_stopping.best_model_weights is not None:
            self.model.load_state_dict(self.early_stopping.best_model_weights)
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.early_stopping.best_model_weights, self.save_path)
            print(f"Best metric model saved to {self.save_path}")

        return self.history
