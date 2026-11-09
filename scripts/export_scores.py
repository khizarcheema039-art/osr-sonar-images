"""
Script: export_scores.py
Loads saved models from outputs/models, executes inference across the test known and unknown sets,
and exports per-sample rejection familiarity scores (higher = more known) to:
frontend/public/data/scores.json

Output schema:
{
    "MSP": {
        "known": [float, ...],
        "unknown": [float, ...]
    },
    "SVDD": {
        "known": [float, ...],
        "unknown": [float, ...]
    },
    "MetricLearning": {
        "known": [float, ...],
        "unknown": [float, ...]
    }
}
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
from configs.config import (
    MODELS_DIR,
    DEVICE,
    NUM_CLASSES,
    UNKNOWN_LABEL,
)
from src.dataset import create_splits, get_dataloaders
from src.models import build_resnet50, EmbeddingNet
from src.methods import MSPDetector, DeepSVDDDetector, DMLDetector


def export_scores(output_path=None):
    if output_path is None:
        output_path = PROJECT_ROOT / "frontend" / "public" / "data" / "scores.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Initializing benchmark dataset splits...")
    splits = create_splits()
    loaders = get_dataloaders(splits)

    # 1. Load Baseline ResNet-50 Checkpoint
    baseline_path = MODELS_DIR / "resnet50_baseline.pth"
    baseline_model = build_resnet50(num_classes=NUM_CLASSES, pretrained=False)
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found at {baseline_path}. "
            "Please train the baseline model first using scripts/train_baseline.py"
        )
    baseline_model.load_state_dict(torch.load(baseline_path, map_location=DEVICE))
    print(f"Loaded baseline checkpoint from {baseline_path}")
    baseline_model.to(DEVICE)
    baseline_model.eval()

    # 2. Load Metric Learning Checkpoint
    metric_path = MODELS_DIR / "metric_embedding_model.pth"
    metric_model = EmbeddingNet(pretrained=False)
    if not metric_path.exists():
        raise FileNotFoundError(
            f"Metric embedding checkpoint not found at {metric_path}. "
            "Please train the metric model first using scripts/train_metric.py"
        )
    metric_model.load_state_dict(torch.load(metric_path, map_location=DEVICE))
    print(f"Loaded metric checkpoint from {metric_path}")
    metric_model.to(DEVICE)
    metric_model.eval()

    results = {}

    # ── Method 1: MSP Thresholding ─────────────────────────────────────────────
    print("\n[1/3] Extracting MSP scores...")
    msp_detector = MSPDetector(baseline_model, device=DEVICE)
    msp_scores, _, true_labels, _ = msp_detector.predict_scores(loaders["test_combined"])
    
    known_mask = true_labels != UNKNOWN_LABEL
    unknown_mask = true_labels == UNKNOWN_LABEL

    results["MSP"] = {
        "known": [round(float(s), 5) for s in msp_scores[known_mask]],
        "unknown": [round(float(s), 5) for s in msp_scores[unknown_mask]],
    }
    print(f"  MSP: {len(results['MSP']['known'])} known, {len(results['MSP']['unknown'])} unknown samples.")

    # ── Method 2: Deep SVDD (Nearest Centroid) ─────────────────────────────────
    print("\n[2/3] Extracting Deep SVDD scores...")
    svdd_detector = DeepSVDDDetector(baseline_model, num_classes=NUM_CLASSES, device=DEVICE)
    train_embs, train_lbls, _ = svdd_detector.extract_features(loaders["train"])
    test_embs, _, _ = svdd_detector.extract_features(loaders["test_combined"])
    svdd_detector.fit(train_embs, train_lbls, fit_svm=False)

    svdd_scores, _ = svdd_detector.compute_scores(test_embs, mode="nearest_centroid")
    results["SVDD"] = {
        "known": [round(float(s), 5) for s in svdd_scores[known_mask]],
        "unknown": [round(float(s), 5) for s in svdd_scores[unknown_mask]],
    }
    print(f"  SVDD: {len(results['SVDD']['known'])} known, {len(results['SVDD']['unknown'])} unknown samples.")

    # ── Method 3: Deep Metric Learning (Triplet Loss) ──────────────────────────
    print("\n[3/3] Extracting Deep Metric Learning scores...")
    dml_detector = DMLDetector(metric_model, num_classes=NUM_CLASSES, device=DEVICE)
    dml_train_embs, dml_train_lbls, _ = dml_detector.extract_embeddings(loaders["train"])
    dml_test_embs, _, _ = dml_detector.extract_embeddings(loaders["test_combined"])
    dml_detector.fit(dml_train_embs, dml_train_lbls)

    dml_scores, _, _ = dml_detector.compute_scores(dml_test_embs)
    results["MetricLearning"] = {
        "known": [round(float(s), 5) for s in dml_scores[known_mask]],
        "unknown": [round(float(s), 5) for s in dml_scores[unknown_mask]],
    }
    print(f"  MetricLearning: {len(results['MetricLearning']['known'])} known, {len(results['MetricLearning']['unknown'])} unknown samples.")

    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSuccessfully exported scores to {output_path}")


if __name__ == "__main__":
    export_scores()
