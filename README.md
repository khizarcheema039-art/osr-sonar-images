# Open-Set Recognition Using Underwater SONAR Images

**Final Year Project (FYP)**  
*Department of Computer Science, Bahria University, Islamabad*  
* **Authors:** Mohammad Khizar Hayat Cheema (01-134231-039), Muhammad Haroon (01-134231-052)  
* **Supervisor:** Dr. Arif Ur Rahman  

---

## 📌 Abstract & Overview

Autonomous underwater vehicles (AUVs) and maritime surveillance systems routinely rely on Forward-Looking Sonar (FLS) to navigate and detect submerged objects. However, standard deep learning classifiers operate under a **closed-set assumption**, presuming every object encountered at test time belongs to a category observed during training. In dynamic underwater environments, novel obstacles, unknown debris, and acoustic anomalies cause conventional classifiers to produce **over-confident, catastrophic misclassifications**.

This repository provides an end-to-end framework for **Open-Set Recognition (OSR)** on underwater sonar imagery. The framework accurately identifies known marine debris categories while reliably detecting and rejecting unknown/novel acoustic targets.

```
+-----------------------------------------------------------------------------------+
|                            SONAR PREPROCESSING PIPELINE                           |
|  Raw Sonar (FLS) ──> RGBA-to-RGB ──> Median Filter (3x3) ──> CLAHE Contrast Norm  |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        RESNET-50 FEATURE EXTRACTION BACKBONE                      |
|  Conv1 + MaxPool ──> ResNet Bottlenecks (Stages 1-4) ──> GAP (2048-D Latent Space)|
+--------------------+----------------------------+--------------------------------+
                     |                            |
       ┌─────────────┴────────────┐               │
       v                          v               v
+──────────────+          +──────────────+  +──────────────────────────────────────+
|   METHOD 1   |          |   METHOD 2   |  |               METHOD 3               |
| MSP Baseline |          |  Deep SVDD   |  |         Deep Metric Learning         |
|   & OpenMax  |          |  (Centroid)  |  |    (Triplet Loss + MultiSimilarity)  |
+──────┬───────+          +──────┬───────+  +──────────────────┬───────────────────+
       │                         │                             │
       └─────────────────────────┼─────────────────────────────┘
                                 v
+-----------------------------------------------------------------------------------+
|                             OSR REJECTION & DECISION                              |
|          Known Category (Class 0..K)   OR   Reject as Novel / Unknown Target      |
+-----------------------------------------------------------------------------------+
```

---

## 📂 Repository Structure

```text
OSR of SONAR images/
├── configs/
│   └── config.py               # Central paths, hyperparameters, class splits & device
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Median noise filtering, CLAHE, and PyTorch augmentations
│   ├── dataset.py              # PyTorch Dataset, disjoint split generation (zero leakage)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── resnet_baseline.py  # ResNet-50 feature extractor and closed-set classifier
│   │   └── embedding_net.py    # Metric learning network with 128-D projection head
│   ├── training/
│   │   ├── __init__.py
│   │   ├── early_stopping.py   # Early stopping utility
│   │   ├── trainer_baseline.py # Training loop for ResNet-50 baseline
│   │   └── trainer_metric.py   # Training loop for Metric Learning (Triplet Loss + Miner)
│   ├── methods/
│   │   ├── __init__.py
│   │   ├── msp.py              # Method 1: Maximum Softmax Probability (MSP)
│   │   ├── openmax.py          # Method 1 (Ext): Extreme Value Theory (EVT) OpenMax
│   │   ├── svdd.py             # Method 2: One-Class SVDD / Centroid Euclidean Distance
│   │   └── dml.py              # Method 3: Deep Metric Learning OSR
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py          # AUROC, AP, TPR, FPR, UDR, ECE, and OSCR metrics
│       └── visualizations.py   # ROC curves, PR curves, t-SNE, and benchmark dashboards
├── scripts/
│   ├── run_eda.py              # Exploratory data analysis & image corruption verification
│   ├── train_baseline.py       # Trains and saves baseline ResNet-50 checkpoint
│   ├── train_metric.py         # Trains and saves Deep Metric Learning model
│   ├── evaluate_all.py         # Executes comprehensive benchmark across all OSR methods
│   └── evaluate_turntable.py   # Phase 2: Cross-domain evaluation on 18 Turntable classes
├── md_fls_dataset/             # Marine Debris FLS dataset (watertank & turntable)
├── outputs/
│   ├── models/                 # Saved model weights (.pth)
│   ├── results/                # Evaluation metric summaries (.json)
│   └── figures/                # Publication-quality benchmark dashboards (.png)
├── requirements.txt            # Python dependencies
└── README.md                   # Complete documentation
```

---

## 🚀 Quickstart & Setup

### 1. Environment Installation
Ensure Python 3.10+ and PyTorch (with CUDA if available) are installed:

```bash
git clone <repo-url>
cd "OSR of SONAR images"
pip install -r requirements.txt
```

### 2. Verify Dataset & Run EDA
Inspect the dataset splits, channel consistency, and acoustic preprocessing pipeline:
```bash
python scripts/run_eda.py
```

### 3. Train Models
Train the baseline ResNet-50 closed-set classifier:
```bash
python scripts/train_baseline.py
```
Train the Deep Metric Learning embedding model:
```bash
python scripts/train_metric.py
```

### 4. Run Comprehensive Benchmark
Evaluate all 3 OSR paradigms (MSP, OpenMax, Deep SVDD, Deep Metric Learning) and generate the master dashboard:
```bash
python scripts/evaluate_all.py
```

### 5. Phase 2: Cross-Domain Turntable Evaluation
Evaluate generalization and unknown rejection on the 18-class Turntable dataset (4,942 images):
```bash
python scripts/evaluate_turntable.py
```

---

## 📊 Benchmark Results

Evaluated on the **Marine Debris Forward-Looking Sonar (MD-FLS)** dataset with disjoint class partitioning:
* **Known Classes ($C_{\text{known}}$)**: `bottle`, `can`, `tire`, `drink-carton`, `hook`, `propeller`
* **Unknown Classes ($C_{\text{unknown}}$)**: `chain`, `valve`, `shampoo-bottle`, `standing-bottle`

| Method | Paradigm | AUROC ↑ | Average Precision (AP) ↑ | TPR ↑ | FPR ↓ | Unknown Detection Rate (UDR) ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MSP Threshold** | Confidence-based | 0.8165 | 0.5782 | 0.8487 | 0.3177 | 0.6823 (68.23%) |
| **Deep SVDD (Centroid)** | Boundary-based | 0.8986 | 0.8275 | 0.8413 | **0.1171** | **0.8829 (88.29%)** |
| **Deep Metric Learning** | Geometric / Embedding | **0.9255** | **0.8448** | **0.9410** | 0.2525 | 0.7475 (74.75%) |

### Key Takeaways:
1. **Deep Metric Learning** with Triplet Loss achieves the **highest overall discrimination (0.9255 AUROC)** and the **highest known true positive rate (94.10% TPR)**.
2. **Deep SVDD (Nearest Centroid)** provides the **safest operating boundary**, yielding the lowest false known rate (**11.71% FPR**) and rejecting **88.29% of unknown objects**.
3. Combining metric embeddings with distance thresholding dramatically outperforms conventional Softmax (MSP) thresholding on sonar acoustic data.

---

## 📜 Citation & Attribution
If you use this codebase or the MD-FLS dataset in your research, please cite:
```bibtex
@inproceedings{valdenegro2025marinedebris,
  title={The Marine Debris Forward-Looking Sonar Datasets},
  author={Valdenegro-Toro, Matias and Padmanabhan, D. C. and Singh, Deepak and Wehbe, Bilal and Petillot, Yvan},
  booktitle={OCEANS 2025 Brest},
  pages={1--10},
  year={2025}
}
```
