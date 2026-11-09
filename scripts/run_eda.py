"""
Script: run_eda.py
Exploratory Data Analysis for the Marine Debris Forward-Looking Sonar (MD-FLS) Dataset.
- Checks image formats and channel counts
- Identifies any corrupted images
- Visualizes class distribution
- Generates raw vs filtered vs CLAHE acoustic preprocessing comparison
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

from configs.config import WATERTANK_DIR, KNOWN_CLASSES, UNKNOWN_CLASSES, FIGURES_DIR
from src.preprocessing import preprocess_sonar_image


def analyze_dataset(dataset_dir=WATERTANK_DIR):
    print(f"Scanning dataset at: {dataset_dir}")
    print("=" * 60)

    class_counts = {}
    corrupted = []
    sizes = set()

    all_classes = KNOWN_CLASSES + UNKNOWN_CLASSES
    for cls_name in all_classes:
        cls_path = Path(dataset_dir) / cls_name
        if not cls_path.exists():
            print(f"Warning: Directory {cls_path} not found.")
            continue

        imgs = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
        class_counts[cls_name] = len(imgs)

        for img_name in imgs:
            full_path = cls_path / img_name
            try:
                with Image.open(full_path) as im:
                    im.verify()
                with Image.open(full_path) as im:
                    sizes.add(im.size)
            except Exception as e:
                corrupted.append((str(full_path), str(e)))

    print(f"Total classes found: {len(class_counts)}")
    print(f"Total images checked: {sum(class_counts.values())}")
    print(f"Corrupted images: {len(corrupted)}")
    print(f"Unique image dimensions: {list(sizes)[:5]}")
    print("-" * 60)

    for cls_name, count in class_counts.items():
        role = "KNOWN" if cls_name in KNOWN_CLASSES else "UNKNOWN"
        print(f"  [{role:7s}] {cls_name:<20} : {count:>4} images")

    # 1. Plot class distribution
    plot_distribution(class_counts)

    # 2. Plot preprocessing comparison
    plot_preprocessing_comparison(dataset_dir)


def plot_distribution(counts):
    fig, ax = plt.subplots(figsize=(12, 6))
    classes = list(counts.keys())
    values = list(counts.values())
    colors = ["steelblue" if c in KNOWN_CLASSES else "tomato" for c in classes]

    bars = ax.bar(classes, values, color=colors, edgecolor="black")
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{bar.get_height()}",
            ha="center",
            fontweight="bold",
            fontsize=9,
        )

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Number of Samples", fontsize=11)
    ax.set_title("Marine Debris FLS Class Distribution (Watertank Dataset)", fontsize=13, fontweight="bold")

    # Custom legend
    import matplotlib.patches as mpatches
    known_patch = mpatches.Patch(color="steelblue", label="Known Classes (Training & Eval)")
    unknown_patch = mpatches.Patch(color="tomato", label="Unknown Classes (Novelty Trigger)")
    ax.legend(handles=[known_patch, unknown_patch], loc="upper right")

    plt.tight_layout()
    save_file = FIGURES_DIR / "class_distribution.png"
    plt.savefig(save_file, dpi=150)
    plt.close()
    print(f"Saved distribution plot to {save_file}")


def plot_preprocessing_comparison(dataset_dir):
    sample_classes = ["bottle", "tire", "chain"]
    fig, axes = plt.subplots(len(sample_classes), 3, figsize=(10, 3 * len(sample_classes)))

    for i, cls_name in enumerate(sample_classes):
        cls_dir = Path(dataset_dir) / cls_name
        imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith((".png", ".jpg"))]
        if not imgs:
            continue
        img_path = cls_dir / imgs[0]

        # Raw
        raw = Image.open(img_path).convert("RGB")
        raw_np = np.array(raw)

        # Median Filtered
        gray = cv2.cvtColor(raw_np, cv2.COLOR_RGB2GRAY)
        med = cv2.medianBlur(gray, 3)

        # CLAHE Enhanced
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(med)

        axes[i, 0].imshow(raw_np)
        axes[i, 0].set_title(f"{cls_name} - Raw Sonar", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(med, cmap="gray")
        axes[i, 1].set_title("Median Denoised (Speckle Filter)", fontsize=10)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(enhanced, cmap="gray")
        axes[i, 2].set_title("CLAHE Contrast Enhanced", fontsize=10)
        axes[i, 2].axis("off")

    plt.suptitle("Sonar Preprocessing Pipeline Stages", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_file = FIGURES_DIR / "eda_preprocessing_stages.png"
    plt.savefig(save_file, dpi=150)
    plt.close()
    print(f"Saved preprocessing stages figure to {save_file}")


if __name__ == "__main__":
    analyze_dataset()
