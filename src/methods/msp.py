"""
Method 1: Maximum Softmax Probability (MSP) Threshold-Based OSR.
Rejects unseen inputs based on classifier prediction certainty.
"""

import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from configs.config import DEVICE, UNKNOWN_LABEL


class MSPDetector:
    """
    Maximum Softmax Probability (MSP) baseline detector.
    Samples with MSP below threshold tau are flagged as unknown.
    """

    def __init__(self, model, device=DEVICE):
        self.model = model.to(device)
        self.device = device
        self.optimal_threshold = 0.5

    def predict_scores(self, loader):
        """
        Extracts max softmax probabilities, predicted classes, true labels, and image paths.
        """
        self.model.eval()
        msp_scores = []
        pred_labels = []
        true_labels = []
        all_paths = []

        with torch.no_grad():
            for images, labels, paths in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)

                max_probs, preds = torch.max(probs, dim=1)

                msp_scores.extend(max_probs.cpu().numpy())
                pred_labels.extend(preds.cpu().numpy())
                true_labels.extend(labels.numpy())
                all_paths.extend(paths)

        return (
            np.array(msp_scores),
            np.array(pred_labels),
            np.array(true_labels),
            all_paths,
        )

    def find_best_threshold(self, msp_scores, true_labels, criterion="youden"):
        """
        Finds optimal rejection threshold tau on test scores.
        Binary label: 1 = Known, 0 = Unknown.
        """
        binary_labels = np.where(true_labels != UNKNOWN_LABEL, 1, 0)
        fpr, tpr, thresholds = roc_curve(binary_labels, msp_scores)

        if criterion == "youden":
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
        elif criterion == "fixed_tpr_95":
            # Closest TPR to 0.95
            best_idx = np.argmin(np.abs(tpr - 0.95))
        else:
            best_idx = len(thresholds) // 2

        self.optimal_threshold = float(thresholds[best_idx])
        return self.optimal_threshold, fpr, tpr, thresholds

    def evaluate(self, msp_scores, pred_labels, true_labels, threshold=None):
        """
        Computes AUROC, AP, TPR, FPR, and UDR given a decision threshold.
        """
        if threshold is None:
            threshold = self.optimal_threshold

        binary_labels = np.where(true_labels != UNKNOWN_LABEL, 1, 0)
        binary_preds = np.where(msp_scores >= threshold, 1, 0)

        # ROC & PR metrics
        fpr_arr, tpr_arr, _ = roc_curve(binary_labels, msp_scores)
        auroc = auc(fpr_arr, tpr_arr)
        ap = average_precision_score(binary_labels, msp_scores)

        # Confusion statistics
        tp = np.sum((binary_preds == 1) & (binary_labels == 1))
        fp = np.sum((binary_preds == 1) & (binary_labels == 0))
        fn = np.sum((binary_preds == 0) & (binary_labels == 1))
        tn = np.sum((binary_preds == 0) & (binary_labels == 0))

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        udr = tn / max(fp + tn, 1)  # Unknown Detection Rate

        return {
            "method": "MSP Threshold",
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
        }
