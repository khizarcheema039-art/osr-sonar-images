"""
Method 2: One-Class Classification & Deep SVDD (Support Vector Data Description).
Constructs class-specific hyperspheres and minimum distance decision boundaries in the ResNet-50 GAP latent space.
"""

import numpy as np
import torch
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve, auc, average_precision_score
from configs.config import DEVICE, UNKNOWN_LABEL, NUM_CLASSES


class DeepSVDDDetector:
    """
    Centroid-based Deep SVDD and One-Class SVM detector operating on 2048-D latent embeddings.
    """

    def __init__(self, model, num_classes=NUM_CLASSES, device=DEVICE):
        self.model = model.to(device)
        self.num_classes = num_classes
        self.device = device
        self.class_centroids = {}
        self.global_centroid = None
        self.oc_svm = None
        self.optimal_threshold = 0.0

    def extract_features(self, loader):
        """
        Extracts L2-normalized 2048-D feature embeddings, labels, and file paths.
        """
        self.model.eval()
        embeddings_list = []
        labels_list = []
        paths_list = []

        with torch.no_grad():
            for images, labels, paths in loader:
                images = images.to(self.device)
                if hasattr(self.model, "extract_features"):
                    feats = self.model.extract_features(images)
                else:
                    feats = self.model(images)

                feats_np = feats.cpu().numpy()
                # L2 normalize
                norms = np.linalg.norm(feats_np, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                norm_feats = feats_np / norms

                embeddings_list.append(norm_feats)
                labels_list.extend(labels.numpy())
                paths_list.extend(paths)

        all_embeddings = np.vstack(embeddings_list)
        all_labels = np.array(labels_list)
        return all_embeddings, all_labels, paths_list

    def fit(self, train_embeddings, train_labels, fit_svm=True):
        """
        Computes class-specific centroids and fits OneClassSVM on training representations.
        """
        self.class_centroids = {}
        for c in range(self.num_classes):
            c_mask = train_labels == c
            if np.sum(c_mask) > 0:
                self.class_centroids[c] = np.mean(train_embeddings[c_mask], axis=0)
                # Re-normalize centroid
                norm = np.linalg.norm(self.class_centroids[c])
                if norm > 0:
                    self.class_centroids[c] /= norm

        self.global_centroid = np.mean(train_embeddings, axis=0)
        norm_g = np.linalg.norm(self.global_centroid)
        if norm_g > 0:
            self.global_centroid /= norm_g

        if fit_svm:
            self.oc_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
            self.oc_svm.fit(train_embeddings)

        print(f"Deep SVDD fitted with {len(self.class_centroids)} class centroids.")

    def compute_scores(self, test_embeddings, mode="nearest_centroid"):
        """
        Computes familiarity score (higher score = more likely to be known).
        """
        if mode == "nearest_centroid":
            # Euclidean distance to nearest centroid
            distances = []
            for emb in test_embeddings:
                dists = [
                    np.linalg.norm(emb - self.class_centroids[c])
                    for c in range(self.num_classes)
                    if c in self.class_centroids
                ]
                distances.append(np.min(dists))
            distances = np.array(distances)
            # Invert: smaller distance -> higher score
            scores = -distances
            return scores, distances

        elif mode == "oc_svm":
            if self.oc_svm is None:
                raise ValueError("OneClassSVM has not been fitted.")
            scores = self.oc_svm.decision_function(test_embeddings)
            return scores, -scores

        else:
            raise ValueError(f"Unknown SVDD mode: {mode}")

    def evaluate(self, test_embeddings, true_labels, mode="nearest_centroid", threshold=None):
        scores, raw_dists = self.compute_scores(test_embeddings, mode=mode)
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
            "method": f"Deep SVDD ({mode})",
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
            "raw_distances": raw_dists,
            "scores": scores,
        }
