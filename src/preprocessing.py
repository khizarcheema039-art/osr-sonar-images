"""
Acoustic Image Preprocessing and Transformation Pipelines for Sonar Imagery.
Includes speckle noise reduction via Median Filtering, contrast enhancement via CLAHE,
and standard PyTorch augmentations.
"""

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from configs.config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    MEDIAN_BLUR_KSIZE,
)


def preprocess_sonar_image(image_input):
    """
    Load and preprocess a single sonar image.
    Handles RGBA/Grayscale -> RGB conversion.
    Applies median filter (speckle noise suppression) + CLAHE (contrast enhancement).

    Args:
        image_input (str, Path, or PIL.Image.Image): Path to image file or PIL image.

    Returns:
        PIL.Image.Image: Preprocessed 3-channel RGB PIL image.
    """
    if isinstance(image_input, (str, bytes)) or hasattr(image_input, "__fspath__"):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    img_np = np.array(img)

    # Convert to grayscale for acoustic filtering
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Step 1: Median filter — removes acoustic speckle noise
    denoised = cv2.medianBlur(gray, MEDIAN_BLUR_KSIZE)

    # Step 2: CLAHE — enhances target-to-background acoustic contrast
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    enhanced = clahe.apply(denoised)

    # Step 3: Convert back to 3-channel RGB for ResNet backbone
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(enhanced_rgb)


def get_train_transforms():
    """
    Geometric and photometric augmentations for training.
    """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_test_transforms():
    """
    Standard deterministic resize and normalization for validation and testing.
    """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
