"""
Evaluation metrics for Closed-Set and Open-Set Recognition.
Includes AUROC, AP, FPR, TPR, UDR, Expected Calibration Error (ECE),
and Open-Set Classification Rate (OSCR).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
from configs.config import UNKNOWN_LABEL


def compute_closed_set_metrics(y_true, y_pred, class_names=None):
    """
    Computes standard closed-set classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
    }


def compute_osr_binary_metrics(true_labels, scores, threshold=None):
    """
    Computes AUROC, AP, TPR, FPR, and UDR on binary labels (1=Known, 0=Unknown).
    """
    binary_labels = np.where(true_labels != UNKNOWN_LABEL, 1, 0)
    fpr_arr, tpr_arr, thresholds = roc_curve(binary_labels, scores)
    auroc = auc(fpr_arr, tpr_arr)
    ap = average_precision_score(binary_labels, scores)

    if threshold is None:
        j_scores = tpr_arr - fpr_arr
        threshold = float(thresholds[np.argmax(j_scores)])

    preds = np.where(scores >= threshold, 1, 0)
    tp = np.sum((preds == 1) & (binary_labels == 1))
    fp = np.sum((preds == 1) & (binary_labels == 0))
    fn = np.sum((preds == 0) & (binary_labels == 1))
    tn = np.sum((preds == 0) & (binary_labels == 0))

    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    udr = tn / max(fp + tn, 1)  # Unknown Detection Rate
    fkr = fp / max(fp + tn, 1)  # False Known Rate (same as FPR)

    return {
        "auroc": auroc,
        "ap": ap,
        "tpr": tpr,
        "fpr": fpr,
        "udr": udr,
        "fkr": fkr,
        "threshold": threshold,
        "fpr_curve": fpr_arr,
        "tpr_curve": tpr_arr,
    }


def compute_ece(confidences, correct, n_bins=10):
    """
    Expected Calibration Error (ECE).
    Measures the alignment between predicted model certainty and actual empirical correctness.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(correct[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_oscr(known_scores, known_correct, unknown_scores):
    """
    Computes Open-Set Classification Rate (OSCR) curve.
    Evaluates Correct Classification Rate (CCR) on knowns versus False Positive Rate (FPR) on unknowns.
    """
    thresholds = np.sort(np.unique(np.concatenate([known_scores, unknown_scores])))[::-1]
    ccr_list = []
    fpr_list = []

    total_known = len(known_scores)
    total_unknown = len(unknown_scores)

    for th in thresholds:
        # Correctly classified knowns whose score is above threshold
        ccr = np.sum((known_scores >= th) & known_correct) / max(total_known, 1)
        # Unknowns erroneously accepted as knowns
        fpr = np.sum(unknown_scores >= th) / max(total_unknown, 1)

        ccr_list.append(ccr)
        fpr_list.append(fpr)

    fpr_arr = np.array(fpr_list)
    ccr_arr = np.array(ccr_list)
    # Sort by FPR for trapezoidal integration
    sort_idx = np.argsort(fpr_arr)
    oscr_auc = np.trapz(ccr_arr[sort_idx], fpr_arr[sort_idx])

    return fpr_arr, ccr_arr, float(oscr_auc)
