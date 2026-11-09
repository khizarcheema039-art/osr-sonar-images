"""
Script: evaluate_all.py
Executes the comprehensive comparative benchmark across all Open-Set Recognition paradigms:
- Method 1: MSP Thresholding
- Method 1 (Ext): OpenMax EVT Modeling
- Method 2: One-Class SVDD (Centroid Euclidean Distance)
- Method 3: Deep Metric Learning (Nearest Centroid)
Generates publication-quality charts and the final FYP master dashboard.
"""

import sys
from pathlib import Path
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
from configs.config import (
    MODELS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    DEVICE,
    NUM_CLASSES,
    UNKNOWN_CLASSES,
    KNOWN_CLASSES,
    IDX_TO_CLASS,
)
from src.dataset import create_splits, get_dataloaders
from src.models import build_resnet50, EmbeddingNet
from src.methods import MSPDetector, OpenMaxDetector, DeepSVDDDetector, DMLDetector
from src.evaluation.metrics import compute_ece
from src.evaluation.visualizations import (
    plot_roc_comparison,
    plot_unknown_detection_per_class,
    plot_tsne_embeddings,
    plot_master_dashboard,
)


def evaluate_all_methods():
    print("Initializing benchmark dataset splits...")
    splits = create_splits()
    loaders = get_dataloaders(splits)

    # 1. Load Baseline Model
    baseline_path = MODELS_DIR / "resnet50_baseline.pth"
    baseline_model = build_resnet50(num_classes=NUM_CLASSES, pretrained=False)
    if baseline_path.exists():
        baseline_model.load_state_dict(torch.load(baseline_path, map_location=DEVICE))
        print(f"Loaded baseline checkpoint from {baseline_path}")
    else:
        print(f"Warning: {baseline_path} not found. Running with initial weights (train first for accurate results).")
    baseline_model.to(DEVICE)
    baseline_model.eval()

    # 2. Load Metric Model
    metric_path = MODELS_DIR / "metric_embedding_model.pth"
    metric_model = EmbeddingNet(pretrained=False)
    if metric_path.exists():
        metric_model.load_state_dict(torch.load(metric_path, map_location=DEVICE))
        print(f"Loaded metric checkpoint from {metric_path}")
    else:
        print(f"Warning: {metric_path} not found. Running with initial weights (train first for accurate results).")
    metric_model.to(DEVICE)
    metric_model.eval()

    all_results = {}
    detection_by_class = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Method 1: MSP Thresholding
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[1/4] Running Method 1: Maximum Softmax Probability (MSP)...")
    msp_detector = MSPDetector(baseline_model, device=DEVICE)
    msp_scores, msp_preds, true_labels, test_paths = msp_detector.predict_scores(loaders["test_combined"])
    msp_th, _, _, _ = msp_detector.find_best_threshold(msp_scores, true_labels)
    msp_res = msp_detector.evaluate(msp_scores, msp_preds, true_labels, threshold=msp_th)
    all_results["MSP Threshold"] = msp_res

    # Per-unknown class detection for MSP
    msp_det = {}
    for u_cls in UNKNOWN_CLASSES:
        mask = [u_cls in p for p in test_paths]
        if np.sum(mask) > 0:
            pct = 100.0 * np.mean(msp_scores[mask] < msp_th)
            msp_det[u_cls] = pct
    detection_by_class["MSP Threshold"] = msp_det

    # ──────────────────────────────────────────────────────────────────────────
    # Method 1 (Ext): OpenMax
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[2/4] Running Method 1 (Ext): OpenMax (EVT)...")
    openmax_detector = OpenMaxDetector(baseline_model, num_classes=NUM_CLASSES, device=DEVICE)
    openmax_detector.fit(loaders["train"])
    om_scores, om_preds, _, _ = openmax_detector.predict_scores(loaders["test_combined"])
    om_res = openmax_detector.evaluate(om_scores, true_labels)
    all_results["OpenMax (EVT)"] = om_res

    om_det = {}
    for u_cls in UNKNOWN_CLASSES:
        mask = [u_cls in p for p in test_paths]
        if np.sum(mask) > 0:
            pct = 100.0 * np.mean(om_scores[mask] < om_res["threshold"])
            om_det[u_cls] = pct
    detection_by_class["OpenMax (EVT)"] = om_det

    # ──────────────────────────────────────────────────────────────────────────
    # Method 2: Deep SVDD (Nearest Centroid)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[3/4] Running Method 2: Deep SVDD (Centroid Distance)...")
    svdd_detector = DeepSVDDDetector(baseline_model, num_classes=NUM_CLASSES, device=DEVICE)
    train_embs, train_lbls, _ = svdd_detector.extract_features(loaders["train"])
    test_embs, _, _ = svdd_detector.extract_features(loaders["test_combined"])
    svdd_detector.fit(train_embs, train_lbls, fit_svm=False)

    svdd_res = svdd_detector.evaluate(test_embs, true_labels, mode="nearest_centroid")
    all_results["Deep SVDD (nearest_centroid)"] = svdd_res

    svdd_det = {}
    for u_cls in UNKNOWN_CLASSES:
        mask = [u_cls in p for p in test_paths]
        if np.sum(mask) > 0:
            pct = 100.0 * np.mean(svdd_res["scores"][mask] < svdd_res["threshold"])
            svdd_det[u_cls] = pct
    detection_by_class["Deep SVDD (nearest_centroid)"] = svdd_det

    # ──────────────────────────────────────────────────────────────────────────
    # Method 3: Deep Metric Learning
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[4/4] Running Method 3: Deep Metric Learning (DML)...")
    dml_detector = DMLDetector(metric_model, num_classes=NUM_CLASSES, device=DEVICE)
    dml_train_embs, dml_train_lbls, _ = dml_detector.extract_embeddings(loaders["train"])
    dml_test_embs, _, _ = dml_detector.extract_embeddings(loaders["test_combined"])
    dml_detector.fit(dml_train_embs, dml_train_lbls)

    dml_res = dml_detector.evaluate(dml_test_embs, true_labels)
    all_results["Deep Metric Learning"] = dml_res

    dml_det = {}
    for u_cls in UNKNOWN_CLASSES:
        mask = [u_cls in p for p in test_paths]
        if np.sum(mask) > 0:
            pct = 100.0 * np.mean(dml_res["scores"][mask] < dml_res["threshold"])
            dml_det[u_cls] = pct
    detection_by_class["Deep Metric Learning"] = dml_det

    # ──────────────────────────────────────────────────────────────────────────
    # Save & Print Benchmark Report
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("FINAL OPEN-SET RECOGNITION BENCHMARK RESULTS")
    print("=" * 75)
    header = f"{'Method':<30} | {'AUROC':>8} | {'AP':>8} | {'TPR':>8} | {'FPR (↓)':>8} | {'UDR (↑)':>8}"
    print(header)
    print("-" * 75)
    for m, r in all_results.items():
        print(f"{m:<30} | {r['auroc']:>8.4f} | {r['ap']:>8.4f} | {r['tpr']:>8.4f} | {r['fpr']:>8.4f} | {r['udr']:>8.4f}")
    print("=" * 75)

    # Save JSON summary (convert numpy arrays to lists)
    serializable = {}
    for m, r in all_results.items():
        serializable[m] = {
            k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
            for k, v in r.items()
            if k not in ["fpr_curve", "tpr_curve", "raw_distances", "scores", "min_distances", "nearest_classes"]
        }
    serializable["detection_by_class"] = detection_by_class

    summary_file = RESULTS_DIR / "benchmark_summary.json"
    with open(summary_file, "w") as f:
        json.dump(serializable, f, indent=4)
    print(f"\nSaved benchmark results to {summary_file}")

    # Generate Dashboards
    print("\nGenerating evaluation dashboards and visualizations...")
    plot_roc_comparison(all_results, save_path=FIGURES_DIR / "all_methods_comparison.png")
    plot_unknown_detection_per_class(detection_by_class, save_path=FIGURES_DIR / "all_methods_unknown_detection.png")
    plot_master_dashboard(all_results, detection_by_class, save_path=FIGURES_DIR / "final_benchmark_dashboard.png")

    # t-SNE on Metric Embeddings
    class_map = {idx: cls_name for idx, cls_name in enumerate(KNOWN_CLASSES)}
    class_map[-1] = "Unknown / Novel"
    plot_tsne_embeddings(dml_test_embs, true_labels, class_map, save_path=FIGURES_DIR / "dml_tsne_visualization.png")

    print("\nAll benchmark artifacts and figures generated successfully!")


if __name__ == "__main__":
    evaluate_all_methods()
