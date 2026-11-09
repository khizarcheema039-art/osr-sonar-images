"""
Master Pipeline Runner for Open-Set Recognition Using Underwater SONAR Images.
Allows running the entire end-to-end pipeline or individual stages with a single command.

Usage:
    python main.py --all                  # Runs entire pipeline from EDA to Turntable testing
    python main.py --eda                  # Runs Exploratory Data Analysis & checks
    python main.py --train-baseline       # Trains Baseline ResNet-50
    python main.py --train-metric         # Trains Deep Metric Learning model
    python main.py --evaluate             # Runs all OSR benchmarks and generates dashboards
    python main.py --turntable            # Runs Phase 2: Turntable cross-domain testing
    python main.py                        # Interactive menu to select what to run
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (
    WATERTANK_DIR,
    TURNTABLE_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    DEVICE,
    KNOWN_CLASSES,
    UNKNOWN_CLASSES,
)


def print_banner():
    banner = """
================================================================================
     OPEN-SET RECOGNITION (OSR) USING UNDERWATER SONAR IMAGES
     Final Year Project — Bahria University, Islamabad
================================================================================
 Device:        {}
 Watertank Dir: {}
 Turntable Dir: {}
 Output Dir:    {}
 Known Classes:   {}
 Unknown Classes: {}
================================================================================
""".format(
        DEVICE,
        WATERTANK_DIR,
        TURNTABLE_DIR,
        PROJECT_ROOT / "outputs",
        KNOWN_CLASSES,
        UNKNOWN_CLASSES,
    )
    print(banner)


def step_eda():
    print("\n" + "=" * 60)
    print("STEP 1: Exploratory Data Analysis & Integrity Verification")
    print("=" * 60)
    from scripts.run_eda import analyze_dataset

    analyze_dataset(WATERTANK_DIR)


def step_train_baseline():
    print("\n" + "=" * 60)
    print("STEP 2: Training Baseline ResNet-50 Classifier")
    print("=" * 60)
    from scripts.train_baseline import main as train_baseline_main

    train_baseline_main()


def step_train_metric():
    print("\n" + "=" * 60)
    print("STEP 3: Training Deep Metric Learning (Triplet Loss + Mining)")
    print("=" * 60)
    from scripts.train_metric import main as train_metric_main

    train_metric_main()


def step_evaluate_all():
    print("\n" + "=" * 60)
    print("STEP 4: Comparative OSR Benchmarks & Master Dashboard")
    print("=" * 60)
    from scripts.evaluate_all import evaluate_all_methods

    evaluate_all_methods()


def step_evaluate_turntable():
    print("\n" + "=" * 60)
    print("STEP 5: Phase 2 — Turntable Cross-Domain Evaluation")
    print("=" * 60)
    from scripts.evaluate_turntable import evaluate_cross_domain

    evaluate_cross_domain()


def run_all_stages():
    start_total = time.time()
    print("\n🚀 Executing Complete End-to-End OSR Pipeline...")

    step_eda()
    step_train_baseline()
    step_train_metric()
    step_evaluate_all()
    step_evaluate_turntable()

    total_time = (time.time() - start_total) / 60
    print("\n" + "=" * 70)
    print(f"🎉 All pipeline stages completed successfully in {total_time:.2f} minutes!")
    print(f"Check your results in: {PROJECT_ROOT / 'outputs'}")
    print("=" * 70)


def interactive_menu():
    print_banner()
    print("Select an option to run:")
    print("  [1] Run COMPLETE Pipeline End-to-End (Steps 1 to 5)")
    print("  [2] Step 1: Run Exploratory Data Analysis & Verify Dataset")
    print("  [3] Step 2: Train Baseline ResNet-50 Model")
    print("  [4] Step 3: Train Deep Metric Learning Model")
    print("  [5] Step 4: Run Comparative OSR Benchmark (MSP, OpenMax, SVDD, DML)")
    print("  [6] Step 5: Run Phase 2 Turntable Cross-Domain Evaluation")
    print("  [0] Exit")
    print("-" * 60)

    try:
        choice = input("Enter choice [0-6] (default: 1): ").strip()
    except EOFError:
        choice = "1"

    if choice in ["1", ""]:
        run_all_stages()
    elif choice == "2":
        step_eda()
    elif choice == "3":
        step_train_baseline()
    elif choice == "4":
        step_train_metric()
    elif choice == "5":
        step_evaluate_all()
    elif choice == "6":
        step_evaluate_turntable()
    elif choice == "0":
        print("Exiting.")
        sys.exit(0)
    else:
        print("Invalid choice. Running complete pipeline by default...")
        run_all_stages()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Master Runner for Open-Set Sonar Image Recognition Pipeline"
    )
    parser.add_argument("--all", action="store_true", help="Execute complete pipeline (all stages)")
    parser.add_argument("--eda", action="store_true", help="Run exploratory data analysis")
    parser.add_argument("--train-baseline", action="store_true", help="Train ResNet-50 baseline")
    parser.add_argument("--train-metric", action="store_true", help="Train Deep Metric Learning model")
    parser.add_argument("--evaluate", action="store_true", help="Run comparative OSR evaluation")
    parser.add_argument("--turntable", action="store_true", help="Run Phase 2 Turntable evaluation")

    return parser.parse_args()


def main():
    args = parse_args()

    # If any specific flags are provided, run those
    if args.all:
        print_banner()
        run_all_stages()
    elif any([args.eda, args.train_baseline, args.train_metric, args.evaluate, args.turntable]):
        print_banner()
        if args.eda:
            step_eda()
        if args.train_baseline:
            step_train_baseline()
        if args.train_metric:
            step_train_metric()
        if args.evaluate:
            step_evaluate_all()
        if args.turntable:
            step_evaluate_turntable()
    else:
        # No arguments given: show interactive menu
        interactive_menu()


if __name__ == "__main__":
    main()
