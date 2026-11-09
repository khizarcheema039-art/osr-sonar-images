"""
Method 1 (Extended): OpenMax Classifier using Extreme Value Theory (EVT).
Models the tail distribution of penultimate layer activations using Weibull distributions
to redistribute softmax probability mass toward an explicit unknown class.
Reference: Bendale & Boult, "Towards Open Set Deep Networks", CVPR 2016.
"""

import numpy as np
import scipy.spatial.distance as spd
from scipy.stats import weibull_min
import torch
from sklearn.metrics import roc_curve, auc, average_precision_score
from configs.config import DEVICE, UNKNOWN_LABEL, NUM_CLASSES


class OpenMaxDetector:
    """
    OpenMax implementation fitting Weibull models to class Mean Activation Vectors (MAVs).
    """

    def __init__(self, model, num_classes=NUM_CLASSES, tailsize=20, alpha=5, device=DEVICE):
        self.model = model.to(device)
        self.num_classes = num_classes
        self.tailsize = tailsize
        self.alpha = min(alpha, num_classes)
        self.device = device
        self.mavs = {}
        self.weibull_models = {}
        self.optimal_threshold = 0.5

    def fit(self, train_loader):
        """
        Computes Mean Activation Vectors (MAVs) and fits Weibull models to distance tails.
        """
        self.model.eval()
        activations_per_class = {c: [] for c in range(self.num_classes)}

        with torch.no_grad():
            for images, labels, _ in train_loader:
                images = images.to(self.device)
                logits = self.model(images).cpu().numpy()
                labels_np = labels.numpy()

                for logit, lbl in zip(logits, labels_np):
                    if 0 <= lbl < self.num_classes:
                        activations_per_class[lbl].append(logit)

        # 1. Compute MAVs
        for c in range(self.num_classes):
            acts = np.array(activations_per_class[c])
            if len(acts) > 0:
                self.mavs[c] = np.mean(acts, axis=0)
            else:
                self.mavs[c] = np.zeros(self.num_classes)

        # 2. Fit Weibull model to tails of distances
        for c in range(self.num_classes):
            acts = np.array(activations_per_class[c])
            if len(acts) < self.tailsize:
                # Fallback if too few samples
                self.weibull_models[c] = None
                continue

            dists = [spd.euclidean(act, self.mavs[c]) for act in acts]
            dists = np.sort(dists)
            tail = dists[-self.tailsize :]

            try:
                # Fit weibull_min with location fixed at 0
                shape, loc, scale = weibull_min.fit(tail, floc=0)
                self.weibull_models[c] = (shape, loc, scale)
            except Exception:
                self.weibull_models[c] = None

        print(f"OpenMax fitted successfully across {self.num_classes} known classes.")

    def _compute_openmax_probs(self, logits):
        """
        Redistributes logits according to Weibull outlier probabilities.
        """
        batch_size = logits.shape[0]
        openmax_probs = np.zeros((batch_size, self.num_classes + 1))

        for i in range(batch_size):
            act = logits[i]
            ranked_classes = np.argsort(act)[::-1]

            omega = np.zeros(self.num_classes)
            for r in range(self.alpha):
                c = ranked_classes[r]
                if self.weibull_models.get(c) is not None:
                    shape, loc, scale = self.weibull_models[c]
                    dist = spd.euclidean(act, self.mavs[c])
                    # Cumulative distribution function
                    w_score = weibull_min.cdf(dist, shape, loc=loc, scale=scale)
                    rank_weight = (self.alpha - r) / self.alpha
                    omega[c] = w_score * rank_weight
                else:
                    omega[c] = 0.0

            # Modified activations
            modified_act = act * (1.0 - omega)
            unknown_act = np.sum(act * omega)

            all_acts = np.append(modified_act, unknown_act)
            # Softmax with numerical stability
            exp_acts = np.exp(all_acts - np.max(all_acts))
            probs = exp_acts / np.sum(exp_acts)
            openmax_probs[i] = probs

        return openmax_probs

    def predict_scores(self, loader):
        """
        Returns:
            known_scores: Probability of being a known class (1 - P(unknown))
            pred_classes: Class with maximum probability
            true_labels: Ground truth labels
            paths: File paths
        """
        self.model.eval()
        known_scores = []
        pred_labels = []
        true_labels = []
        all_paths = []

        with torch.no_grad():
            for images, labels, paths in loader:
                images = images.to(self.device)
                logits = self.model(images).cpu().numpy()

                openmax_probs = self._compute_openmax_probs(logits)

                # Probability of being known = max over known class probabilities
                known_prob = np.max(openmax_probs[:, : self.num_classes], axis=1)
                preds = np.argmax(openmax_probs[:, : self.num_classes], axis=1)

                known_scores.extend(known_prob)
                pred_labels.extend(preds)
                true_labels.extend(labels.numpy())
                all_paths.extend(paths)

        return (
            np.array(known_scores),
            np.array(pred_labels),
            np.array(true_labels),
            all_paths,
        )

    def evaluate(self, known_scores, true_labels, threshold=None):
        binary_labels = np.where(true_labels != UNKNOWN_LABEL, 1, 0)
        fpr_arr, tpr_arr, thresholds = roc_curve(binary_labels, known_scores)
        auroc = auc(fpr_arr, tpr_arr)
        ap = average_precision_score(binary_labels, known_scores)

        if threshold is None:
            j_scores = tpr_arr - fpr_arr
            threshold = float(thresholds[np.argmax(j_scores)])

        self.optimal_threshold = threshold
        binary_preds = np.where(known_scores >= threshold, 1, 0)

        tp = np.sum((binary_preds == 1) & (binary_labels == 1))
        fp = np.sum((binary_preds == 1) & (binary_labels == 0))
        fn = np.sum((binary_preds == 0) & (binary_labels == 1))
        tn = np.sum((binary_preds == 0) & (binary_labels == 0))

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        udr = tn / max(fp + tn, 1)

        return {
            "method": "OpenMax (EVT)",
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
