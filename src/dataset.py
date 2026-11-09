"""
Sonar Dataset Loader and Disjoint Open-Set Split Generator.
Guarantees zero data leakage and proper disjoint partitioning between known and unknown classes.
"""

import os
import random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from configs.config import (
    WATERTANK_DIR,
    KNOWN_CLASSES,
    UNKNOWN_CLASSES,
    CLASS_TO_IDX,
    UNKNOWN_LABEL,
    BATCH_SIZE,
    NUM_WORKERS,
    RANDOM_SEED,
)
from src.preprocessing import (
    preprocess_sonar_image,
    get_train_transforms,
    get_val_test_transforms,
)


class SonarDataset(Dataset):
    """
    PyTorch Dataset for Sonar Images.
    Handles RGBA to RGB conversion and acoustic noise filtering on-the-fly.
    """

    def __init__(self, data_list, transform=None, preprocess=True):
        """
        Args:
            data_list (list): List of (image_path, label) tuples.
                              Label is >= 0 for known classes, -1 for unknown.
            transform (callable, optional): PyTorch torchvision transform.
            preprocess (bool): If True, applies speckle noise median filter + CLAHE.
        """
        self.data_list = data_list
        self.transform = transform
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        img_path_str = str(img_path)

        if self.preprocess:
            img = preprocess_sonar_image(img_path_str)
        else:
            img = Image.open(img_path_str).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label, img_path_str


def create_splits(
    dataset_path=WATERTANK_DIR,
    known_classes=KNOWN_CLASSES,
    unknown_classes=UNKNOWN_CLASSES,
    class_to_idx=CLASS_TO_IDX,
    train_ratio=0.70,
    val_ratio=0.15,
    seed=RANDOM_SEED,
):
    """
    Split known classes into train (70%), val (15%), test (15%).
    Unknown classes are 100% placed into test_unknown.

    Returns:
        dict: {
            'train': [(path, label), ...],
            'val': [(path, label), ...],
            'test_known': [(path, label), ...],
            'test_unknown': [(path, label), ...]
        }
    """
    dataset_path = Path(dataset_path)
    random.seed(seed)
    np.random.seed(seed)

    train_list = []
    val_list = []
    test_known = []
    test_unknown = []

    # ── Known Classes Splitting ───────────────────────────────────────────────
    for cls_name in known_classes:
        cls_dir = dataset_path / cls_name
        if not cls_dir.exists():
            continue

        imgs = sorted([
            str(cls_dir / f)
            for f in os.listdir(cls_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ])

        random.shuffle(imgs)
        n_total = len(imgs)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        cls_train = imgs[:n_train]
        cls_val = imgs[n_train : n_train + n_val]
        cls_test = imgs[n_train + n_val :]

        label = class_to_idx[cls_name]
        train_list.extend([(p, label) for p in cls_train])
        val_list.extend([(p, label) for p in cls_val])
        test_known.extend([(p, label) for p in cls_test])

    # ── Unknown Classes (100% into test) ──────────────────────────────────────
    for cls_name in unknown_classes:
        cls_dir = dataset_path / cls_name
        if not cls_dir.exists():
            continue

        imgs = sorted([
            str(cls_dir / f)
            for f in os.listdir(cls_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ])

        test_unknown.extend([(p, UNKNOWN_LABEL) for p in imgs])

    # ── Leakage Verification ──────────────────────────────────────────────────
    train_set = {p for p, _ in train_list}
    val_set = {p for p, _ in val_list}
    test_k_set = {p for p, _ in test_known}
    test_u_set = {p for p, _ in test_unknown}

    leakage_train_val = len(train_set & val_set)
    leakage_train_test = len(train_set & test_k_set)
    leakage_val_test = len(val_set & test_k_set)
    overlap_known_unknown = len((train_set | val_set | test_k_set) & test_u_set)

    if any([leakage_train_val, leakage_train_test, leakage_val_test, overlap_known_unknown]):
        raise RuntimeError(
            f"Data leakage detected! "
            f"train∩val={leakage_train_val}, train∩test={leakage_train_test}, "
            f"val∩test={leakage_val_test}, known∩unknown={overlap_known_unknown}"
        )

    return {
        "train": train_list,
        "val": val_list,
        "test_known": test_known,
        "test_unknown": test_unknown,
    }


def get_dataloaders(splits, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Creates PyTorch DataLoaders for train, val, test_known, test_unknown,
    and a combined test loader (test_known + test_unknown).
    """
    train_transform = get_train_transforms()
    eval_transform = get_val_test_transforms()

    train_ds = SonarDataset(splits["train"], transform=train_transform, preprocess=True)
    val_ds = SonarDataset(splits["val"], transform=eval_transform, preprocess=True)
    test_k_ds = SonarDataset(splits["test_known"], transform=eval_transform, preprocess=True)
    test_u_ds = SonarDataset(splits["test_unknown"], transform=eval_transform, preprocess=True)

    combined_test_list = splits["test_known"] + splits["test_unknown"]
    test_combined_ds = SonarDataset(combined_test_list, transform=eval_transform, preprocess=True)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test_known": DataLoader(test_k_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test_unknown": DataLoader(test_u_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test_combined": DataLoader(test_combined_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    return loaders
