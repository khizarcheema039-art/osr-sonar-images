"""
Method 3: Deep Metric Learning (DML) Open-Set Recognition.
Measures Euclidean/Cosine distance in the 128-D metric embedding space optimized with Triplet Loss.
"""

import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, average_precision_score
from configs.config import DEVICE, UNKNOWN_LABEL, NUM_CLASSES


class DMLDetector:
    """
    Open-Set Recognition based on nearest centroid distance in the Deep Metric space.
    """

    def __init__(self, metric_model, num_classes=NUM_CLASSES, device=DEVICE):
        self.model = metric_model.to(device)
        self.num_classes = num_classes
        self.device = device
        self.metric_centroids = {}
        self.optimal_threshold = 0.0

    def extract_embeddings(self, loader):
        """
        Extracts L2-normalized 128-D metric embeddings, labels, and file paths.
        """
        self.model.eval()
        embeddings_list = []
        labels_list = []
        paths_list = []

        with torch.no_grad():
            for images, labels, paths in loader:
                images = images.to(self.device)
                embs = self.model(images)

                embeddings_list.append(embs.cpu().numpy())
                labels_list.extend(labels.numpy())
                paths_list.extend(paths)

        all_embeddings = np.vstack(embeddings_list)
        all_labels = np.array(labels_list)
        return all_embeddings, all_labels, paths_list

    def fit(self, train_embeddings, train_labels):
        """
        Calculates normalized class centroids in metric space.
        """
        self.metric_centroids = {}
        for c in range(self.num_classes):
            mask = train_labels == c
            if np.sum(mask) > 0:
                centroid = np.mean(train_embeddings[mask], axis=0)
                # L2 re-normalization
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid /= norm
                self.metric_centroids[c] = centroid

        print(f"DML fitted with {len(self.metric_centroids)} class centroids in metric space.")

    def compute_scores(self, test_embeddings):
        """
        Computes distance to nearest centroid.
        Returns:
            scores: Higher score = more likely to belong to a known class (-min_dist)
            min_dists: Minimum Euclidean distance to any known centroid
            nearest_classes: ID of the closest known class
        """
        min_dists = []
        nearest_classes = []

        for emb in test_embeddings:
            dists = [
                np.linalg.norm(emb - self.metric_centroids[c])
                for c in range(self.num_classes)
                if c in self.metric_centroids
            ]
            min_idx = np.argmin(dists)
            min_dists.append(dists[min_idx])
            nearest_classes.append(min_idx)

        min_dists = np.array(min_dists)
        nearest_classes = np.array(nearest_classes)
        # Score: negative distance (or similarity)
        scores = -min_dists
        return scores, min_dists, nearest_classes

    def evaluate(self, test_embeddings, true_labels, threshold=None):
        scores, min_dists, nearest_classes = self.compute_scores(test_embeddings)
        binary_labels = np.where(true_labels != UNKNOWN_LABEL, 1, 0)

        fpr_arr, tpr_arr, thresholds = roc_curve(binary_labels, scores)
        auroc = auc(fpr_arr, tpr_arr)
        ap = average_precision_score(binary_labels, scores)

        if threshold is None:
            j_scores = tpr_arr - fpr_arr
            threshold = float(thresholds[np.argmax(j_scores)])

        self.optimal_threshold = threshold
        binary_preds = np.where(scores >= threshold, 1, 0)

        tp = np.sum((binary_preds == 1) & (binary_labels == 1))
        fp = np.sum((binary_preds == 1) & (binary_labels == 0))
        fn = np.sum((binary_preds == 0) & (binary_labels == 1))
        tn = np.sum((binary_preds == 0) & (binary_labels == 0))

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        udr = tn / max(fp + tn, 1)

        return {
            "method": "Deep Metric Learning",
            "threshold": threshold,
            "auroc": auroc,
            "ap": ap,
            "tpr": tpr,
            "fpr": fpr,
            "udr": udr,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "fpr_curve": fpr_arr,
            "tpr_curve": tpr_arr,
            "min_distances": min_dists,
            "nearest_classes": nearest_classes,
            "scores": scores,
        }
