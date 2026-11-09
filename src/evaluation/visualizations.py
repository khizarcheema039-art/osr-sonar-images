"""
Visualization and Dashboard Generation for Open-Set Sonar Recognition.
Generates all 22+ publication-quality figures matching the original notebook:
- EDA & preprocessing figures
- Training curves and confusion matrices
- Method-specific dashboards (MSP, SVDD, DML)
- Benchmark comparisons, ROC/PR curves, t-SNE embeddings, and Master Dashboard
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from configs.config import FIGURES_DIR, UNKNOWN_CLASSES, KNOWN_CLASSES

sns.set_theme(style="whitegrid")


# ── 1. Training Curves & Confusion Matrix ─────────────────────────────────────
def plot_training_curves(history, model_name="resnet50_baseline", save_path=FIGURES_DIR / "resnet50_baseline_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", lw=2)
    axes[0].plot(epochs, history["val_loss"], "r-s", label="Val Loss", lw=2)
    axes[0].set_title(f"{model_name} — Loss History", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Acc (%)", lw=2)
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]], "r-s", label="Val Acc (%)", lw=2)
    axes[1].set_title(f"{model_name} — Accuracy History", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_confusion_matrix_baseline(y_true, y_pred, class_names=KNOWN_CLASSES, save_path=FIGURES_DIR / "confusion_matrix_baseline.png"):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("Baseline Closed-Set Confusion Matrix (Normalized)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


# ── 2. MSP Complete Dashboard ────────────────────────────────────────────────
def plot_msp_dashboard(known_scores, unknown_scores, fpr, tpr, auroc, threshold, save_path=FIGURES_DIR / "msp_complete_dashboard.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Distribution
    sns.kdeplot(known_scores, fill=True, color="steelblue", label="Known Classes", ax=axes[0])
    sns.kdeplot(unknown_scores, fill=True, color="tomato", label="Unknown Classes", ax=axes[0])
    axes[0].axvline(threshold, color="black", linestyle="--", label=f"Threshold tau={threshold:.3f}")
    axes[0].set_title("MSP Probability Distributions", fontsize=12, fontweight="bold")
    axes[0].legend()

    # ROC Curve
    axes[1].plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUROC = {auroc:.4f})")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].set_title("MSP ROC Curve", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].legend()

    # Score Boxplot
    data = [known_scores, unknown_scores]
    axes[2].boxplot(data, tick_labels=["Known", "Unknown"], patch_artist=True)
    axes[2].set_title("MSP Score Quartiles", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Confidence Score")

    plt.suptitle("Method 1: Maximum Softmax Probability (MSP) Dashboard", fontsize=14, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved MSP dashboard to {save_path}")


# ── 3. SVDD Complete Dashboard ───────────────────────────────────────────────
def plot_svdd_dashboard(known_dists, unknown_dists, fpr, tpr, auroc, threshold, save_path=FIGURES_DIR / "svdd_complete_dashboard.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.kdeplot(known_dists, fill=True, color="green", label="Known Classes (Compact)", ax=axes[0])
    sns.kdeplot(unknown_dists, fill=True, color="red", label="Unknown Classes (Outliers)", ax=axes[0])
    axes[0].axvline(threshold, color="black", linestyle="--", label=f"Threshold tau={threshold:.3f}")
    axes[0].set_title("Centroid Distance Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Euclidean Distance to Nearest Centroid")
    axes[0].legend()

    axes[1].plot(fpr, tpr, color="orange", lw=2, label=f"ROC (AUROC = {auroc:.4f})")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].set_title("Deep SVDD ROC Curve", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].legend()

    axes[2].boxplot([known_dists, unknown_dists], tick_labels=["Known", "Unknown"], patch_artist=True)
    axes[2].set_title("Latent Distance Spread", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Distance")

    plt.suptitle("Method 2: Deep SVDD (Centroid Euclidean Distance) Dashboard", fontsize=14, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SVDD dashboard to {save_path}")


# ── 4. DML Complete Dashboard ────────────────────────────────────────────────
def plot_dml_dashboard(known_dists, unknown_dists, fpr, tpr, auroc, threshold, save_path=FIGURES_DIR / "dml_complete_dashboard.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.kdeplot(known_dists, fill=True, color="steelblue", label="Known (Metric Clustered)", ax=axes[0])
    sns.kdeplot(unknown_dists, fill=True, color="tomato", label="Unknown (Metric Separated)", ax=axes[0])
    axes[0].axvline(threshold, color="black", linestyle="--", label=f"Threshold tau={threshold:.3f}")
    axes[0].set_title("Metric Latent Distance Distribution", fontsize=12, fontweight="bold")
    axes[0].legend()

    axes[1].plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUROC = {auroc:.4f})")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].set_title("DML ROC Curve", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].legend()

    axes[2].boxplot([known_dists, unknown_dists], tick_labels=["Known", "Unknown"], patch_artist=True)
    axes[2].set_title("Metric Distance Boxplot", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Euclidean Distance in 128-D Space")

    plt.suptitle("Method 3: Deep Metric Learning (Triplet Loss) Dashboard", fontsize=14, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved DML dashboard to {save_path}")


# ── 5. Comparative Benchmarks ─────────────────────────────────────────────────
def plot_roc_comparison(results_dict, save_path=FIGURES_DIR / "all_methods_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    colors = {
        "MSP Threshold": "tomato",
        "Deep SVDD (nearest_centroid)": "orange",
        "Deep Metric Learning": "steelblue",
        "OpenMax (EVT)": "green",
    }

    for name, res in results_dict.items():
        color = colors.get(name, "purple")
        axes[0].plot(
            res["fpr_curve"],
            res["tpr_curve"],
            label=f"{name} (AUROC = {res['auroc']:.4f})",
            linewidth=2,
            color=color,
        )
        axes[0].scatter(
            res["fpr"],
            res["tpr"],
            color=color,
            s=90,
            zorder=5,
            edgecolors="black",
        )

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Chance")
    axes[0].set_title("ROC Curves Comparison", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (FPR)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (TPR)", fontsize=11)
    axes[0].legend(loc="lower right", fontsize=10)

    method_labels = list(results_dict.keys())
    aurocs = [results_dict[m]["auroc"] for m in method_labels]
    udrs = [results_dict[m]["udr"] for m in method_labels]
    fprs = [results_dict[m]["fpr"] for m in method_labels]

    x = np.arange(len(method_labels))
    w = 0.25
    short_labels = [m.replace(" (nearest_centroid)", "") for m in method_labels]

    bars1 = axes[1].bar(x - w, aurocs, w, label="AUROC", color="steelblue")
    bars2 = axes[1].bar(x, udrs, w, label="UDR", color="forestgreen", alpha=0.85)
    bars3 = axes[1].bar(x + w, fprs, w, label="FPR (lower=better)", color="tomato", alpha=0.85)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_labels, rotation=15, fontsize=10)
    axes[1].set_ylim(0, 1.15)
    axes[1].set_ylabel("Score", fontsize=11)
    axes[1].set_title("Key Performance Metrics", fontsize=13, fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=10)

    plt.suptitle("Complete Open-Set Recognition Benchmark", fontsize=15, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC comparison to {save_path}")


def plot_unknown_detection_per_class(
    detection_by_class, save_path=FIGURES_DIR / "all_methods_unknown_detection.png"
):
    fig, ax = plt.subplots(figsize=(12, 6))
    methods = list(detection_by_class.keys())
    classes = UNKNOWN_CLASSES

    x = np.arange(len(classes))
    total_m = len(methods)
    w = 0.8 / total_m
    colors = ["tomato", "orange", "steelblue", "green"]

    for i, m in enumerate(methods):
        values = [detection_by_class[m].get(c, 0.0) for c in classes]
        offset = (i - total_m / 2 + 0.5) * w
        bars = ax.bar(x + offset, values, w, label=m, color=colors[i % len(colors)], alpha=0.85)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.0,
                f"{bar.get_height():.1f}%",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Detection / Rejection Rate (%)", fontsize=11)
    ax.set_title("Unknown Rejection Rate per Object Category", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved unknown detection plot to {save_path}")


def plot_tsne_embeddings(
    embeddings, labels, class_names, save_path=FIGURES_DIR / "dml_tsne_visualization.png"
):
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    feats_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for lbl, color in zip(unique_labels, colors):
        mask = labels == lbl
        name = class_names.get(lbl, "Unknown" if lbl == -1 else f"Class {lbl}")
        alpha = 0.85 if lbl != -1 else 0.4
        marker = "o" if lbl != -1 else "x"
        ax.scatter(
            feats_2d[mask, 0],
            feats_2d[mask, 1],
            label=name,
            c=[color],
            alpha=alpha,
            marker=marker,
            s=40 if lbl != -1 else 30,
        )

    ax.set_title("t-SNE Latent Space Representation", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved t-SNE plot to {save_path}")


def plot_master_dashboard(
    results_dict, detection_by_class, save_path=FIGURES_DIR / "final_benchmark_dashboard.png"
):
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        "Open-Set Recognition Using Underwater SONAR Images — FYP Benchmark\n"
        "Comparative Evaluation: MSP vs Deep SVDD vs Deep Metric Learning",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    methods = list(results_dict.keys())
    short_methods = [m.replace(" (nearest_centroid)", "") for m in methods]
    colors = ["tomato", "orange", "steelblue", "forestgreen"][: len(methods)]

    # 1. AUROC
    ax1 = fig.add_subplot(2, 3, 1)
    aurocs = [results_dict[m]["auroc"] for m in methods]
    bars = ax1.bar(short_methods, aurocs, color=colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, aurocs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.4f}", ha="center", fontweight="bold", fontsize=10)
    ax1.set_ylim(0.7, 1.0)
    ax1.set_ylabel("AUROC", fontsize=11)
    ax1.set_title("AUROC (Higher = Better)", fontweight="bold")

    # 2. FPR
    ax2 = fig.add_subplot(2, 3, 2)
    fprs = [results_dict[m]["fpr"] for m in methods]
    bars = ax2.bar(short_methods, fprs, color=colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, fprs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.4f}", ha="center", fontweight="bold", fontsize=10)
    ax2.set_ylim(0, 0.45)
    ax2.set_ylabel("FPR", fontsize=11)
    ax2.set_title("False Positive Rate (Lower = Better)", fontweight="bold")

    # 3. UDR
    ax3 = fig.add_subplot(2, 3, 3)
    udrs = [results_dict[m]["udr"] for m in methods]
    bars = ax3.bar(short_methods, udrs, color=colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, udrs):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.4f}", ha="center", fontweight="bold", fontsize=10)
    ax3.set_ylim(0.5, 1.05)
    ax3.set_ylabel("UDR", fontsize=11)
    ax3.set_title("Unknown Detection Rate (Higher = Better)", fontweight="bold")

    # 4. ROC Curves
    ax4 = fig.add_subplot(2, 3, 4)
    for m, c in zip(methods, colors):
        ax4.plot(results_dict[m]["fpr_curve"], results_dict[m]["tpr_curve"], color=c, lw=2, label=f"{m} ({results_dict[m]['auroc']:.4f})")
    ax4.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax4.set_title("ROC Curves", fontweight="bold")
    ax4.set_xlabel("FPR")
    ax4.set_ylabel("TPR")
    ax4.legend(fontsize=8, loc="lower right")

    # 5. Per-Class Unknown Detection
    ax5 = fig.add_subplot(2, 3, 5)
    x = np.arange(len(UNKNOWN_CLASSES))
    w = 0.8 / len(methods)
    for i, (m, c) in enumerate(zip(methods, colors)):
        vals = [detection_by_class[m].get(cls_name, 0.0) for cls_name in UNKNOWN_CLASSES]
        offset = (i - len(methods) / 2 + 0.5) * w
        ax5.bar(x + offset, vals, w, label=short_methods[i], color=c, alpha=0.85)
    ax5.set_xticks(x)
    ax5.set_xticklabels(UNKNOWN_CLASSES, fontsize=9, rotation=15)
    ax5.set_ylim(0, 115)
    ax5.set_ylabel("Detection %")
    ax5.set_title("Unknown Detection per Class", fontweight="bold")
    ax5.legend(fontsize=8)

    # 6. Benchmark Summary Table
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    table_data = [["Metric"] + short_methods]
    metrics_to_show = [("AUROC", "auroc"), ("Avg Precision", "ap"), ("TPR", "tpr"), ("FPR (↓)", "fpr"), ("UDR (↑)", "udr")]
    for label, key in metrics_to_show:
        row = [label] + [f"{results_dict[m][key]:.4f}" for m in methods]
        table_data.append(row)

    table = ax6.table(cellText=table_data[1:], colLabels=table_data[0], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    for j in range(len(table_data[0])):
        table[0, j].set_facecolor("steelblue")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax6.set_title("Final Benchmark Table", fontweight="bold", pad=20)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved master dashboard to {save_path}")
