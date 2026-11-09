"""
Script: evaluate_turntable.py
Phase 2: Cross-Domain & Out-of-Distribution Evaluation on Turntable Sonar Dataset.
Tests the generalizability and open-set rejection capabilities of models trained on Watertank
when deployed against 18 Turntable classes (including 360-degree yaw variations and completely unseen objects).
"""

import sys
from pathlib import Path
import os
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from configs.config import (
    TURNTABLE_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    DEVICE,
    NUM_CLASSES,
    KNOWN_CLASSES,
    BATCH_SIZE,
    UNKNOWN_LABEL,
)
from src.dataset import SonarDataset
from src.preprocessing import get_val_test_transforms
from src.models import build_resnet50, EmbeddingNet
from src.methods import MSPDetector, DeepSVDDDetector, DMLDetector


def prepare_turntable_data():
    """
    Scans Turntable dataset and maps classes into:
    - Known counterparts (if equivalent)
    - Completely novel / unknown classes (e.g. wrench, metal-box, plastic-bidon, etc.)
    """
    turntable_path = Path(TURNTABLE_DIR)
    if not turntable_path.exists():
        raise FileNotFoundError(f"Turntable directory {turntable_path} does not exist.")

    all_dirs = sorted([d for d in os.listdir(turntable_path) if (turntable_path / d).is_dir()])

    # Mapping turntable class names to known labels if direct match, else unknown (-1)
    # Direct correspondences:
    KNOWN_MAPPING = {
        "can": 1,  # 'can' index in KNOWN_CLASSES
        "drink-carton": 3,
        "plastic-propeller": 5,
        "large-tire": 2,
        "small-tire": 2,
        "glass-bottle": 0,
        "plastic-bottle": 0,
        "brown-glass-bottle": 0,
    }

    samples = []
    class_summary = {}

    for cls_name in all_dirs:
        dir_path = turntable_path / cls_name
        imgs = [
            str(dir_path / f)
            for f in os.listdir(dir_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
        label = KNOWN_MAPPING.get(cls_name, UNKNOWN_LABEL)
        class_summary[cls_name] = {
            "count": len(imgs),
            "assigned_label": "KNOWN" if label != UNKNOWN_LABEL else "NOVEL_UNKNOWN",
        }
        for img_path in imgs:
            samples.append((img_path, label))

    print("Turntable Dataset Categorization:")
    print("=" * 60)
    for c, info in class_summary.items():
        print(f"  {c:<25} : {info['count']:>4} imgs [{info['assigned_label']}]")
    print(f"Total Turntable Images: {len(samples)}")
    print("=" * 60)

    transform = get_val_test_transforms()
    dataset = SonarDataset(samples, transform=transform, preprocess=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return loader, samples, class_summary


def evaluate_cross_domain():
    print("Initializing Phase 2: Turntable Cross-Domain OSR Evaluation...")
    loader, samples, class_summary = prepare_turntable_data()

    # Load trained models
    baseline_path = MODELS_DIR / "resnet50_baseline.pth"
    metric_path = MODELS_DIR / "metric_embedding_model.pth"

    baseline_model = build_resnet50(num_classes=NUM_CLASSES, pretrained=False)
    if baseline_path.exists():
        baseline_model.load_state_dict(torch.load(baseline_path, map_location=DEVICE))
    baseline_model.to(DEVICE)
    baseline_model.eval()

    metric_model = EmbeddingNet(pretrained=False)
    if metric_path.exists():
        metric_model.load_state_dict(torch.load(metric_path, map_location=DEVICE))
    metric_model.to(DEVICE)
    metric_model.eval()

    # Detectors
    msp_detector = MSPDetector(baseline_model, device=DEVICE)
    svdd_detector = DeepSVDDDetector(baseline_model, num_classes=NUM_CLASSES, device=DEVICE)
    dml_detector = DMLDetector(metric_model, num_classes=NUM_CLASSES, device=DEVICE)

    # Load Watertank centroids for SVDD & DML
    from src.dataset import create_splits, get_dataloaders
    train_splits = create_splits()
    train_loaders = get_dataloaders(train_splits)

    print("Fitting centroids on Watertank training features...")
    train_feats, train_lbls, _ = svdd_detector.extract_features(train_loaders["train"])
    svdd_detector.fit(train_feats, train_lbls, fit_svm=False)

    train_embs, train_lbls, _ = dml_detector.extract_embeddings(train_loaders["train"])
    dml_detector.fit(train_embs, train_lbls)

    # Evaluate on Turntable
    print("\nRunning inference on Turntable dataset...")
    msp_scores, _, true_labels, paths = msp_detector.predict_scores(loader)
    tt_feats, _, _ = svdd_detector.extract_features(loader)
    tt_embs, _, _ = dml_detector.extract_embeddings(loader)

    svdd_scores, _ = svdd_detector.compute_scores(tt_feats)
    dml_scores, _, _ = dml_detector.compute_scores(tt_embs)

    # Evaluate OSR metrics on Turntable
    msp_eval = msp_detector.evaluate(msp_scores, np.zeros_like(true_labels), true_labels)
    svdd_eval = svdd_detector.evaluate(tt_feats, true_labels)
    dml_eval = dml_detector.evaluate(tt_embs, true_labels)

    print("\n" + "=" * 70)
    print("PHASE 2 (TURNTABLE) CROSS-DOMAIN OSR PERFORMANCE")
    print("=" * 70)
    print(f"{'Method':<25} | {'AUROC':>8} | {'AP':>8} | {'UDR (Rejection)':>15}")
    print("-" * 70)
    print(f"{'MSP Threshold':<25} | {msp_eval['auroc']:>8.4f} | {msp_eval['ap']:>8.4f} | {msp_eval['udr'] * 100:>14.2f}%")
    print(f"{'Deep SVDD':<25} | {svdd_eval['auroc']:>8.4f} | {svdd_eval['ap']:>8.4f} | {svdd_eval['udr'] * 100:>14.2f}%")
    print(f"{'Deep Metric Learning':<25} | {dml_eval['auroc']:>8.4f} | {dml_eval['ap']:>8.4f} | {dml_eval['udr'] * 100:>14.2f}%")
    print("=" * 70)

    # Per novel class rejection rate
    novel_classes = [c for c, info in class_summary.items() if info["assigned_label"] == "NOVEL_UNKNOWN"]
    rejection_per_class = {}

    for nc in novel_classes:
        mask = np.array([f"/{nc}/" in p.replace("\\", "/") for p in paths])
        if np.sum(mask) > 0:
            rejection_per_class[nc] = {
                "MSP": float(100.0 * np.mean(msp_scores[mask] < msp_eval["threshold"])),
                "SVDD": float(100.0 * np.mean(svdd_scores[mask] < svdd_eval["threshold"])),
                "DML": float(100.0 * np.mean(dml_scores[mask] < dml_eval["threshold"])),
            }

    # Plot Turntable Novel Class Rejection
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(novel_classes))
    w = 0.25
    msp_vals = [rejection_per_class[c]["MSP"] for c in novel_classes]
    svdd_vals = [rejection_per_class[c]["SVDD"] for c in novel_classes]
    dml_vals = [rejection_per_class[c]["DML"] for c in novel_classes]

    ax.bar(x - w, msp_vals, w, label="MSP Threshold", color="tomato")
    ax.bar(x, svdd_vals, w, label="Deep SVDD", color="orange")
    ax.bar(x + w, dml_vals, w, label="Deep Metric Learning", color="steelblue")

    ax.set_xticks(x)
    ax.set_xticklabels(novel_classes, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Rejection Rate (%)", fontsize=11)
    ax.set_title("Cross-Domain Novel Class Rejection (Turntable Dataset)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend()
    plt.tight_layout()

    save_fig = FIGURES_DIR / "turntable_novel_rejection.png"
    plt.savefig(save_fig, dpi=150)
    plt.close()
    print(f"Saved Turntable rejection plot to {save_fig}")

    # Save results JSON
    save_json = RESULTS_DIR / "turntable_benchmark_summary.json"
    with open(save_json, "w") as f:
        json.dump(
            {
                "msp": {k: float(v) for k, v in msp_eval.items() if isinstance(v, (float, np.floating))},
                "svdd": {k: float(v) for k, v in svdd_eval.items() if isinstance(v, (float, np.floating))},
                "dml": {k: float(v) for k, v in dml_eval.items() if isinstance(v, (float, np.floating))},
                "rejection_per_class": rejection_per_class,
            },
            f,
            indent=4,
        )
    print(f"Saved Turntable benchmark results to {save_json}")


if __name__ == "__main__":
    evaluate_cross_domain()
