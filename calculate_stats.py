"""
Calculate mean and standard deviation for training dataset images.
This script iterates through training images to compute dataset statistics
for proper normalization.

Usage:
    python calculate_stats.py                    # Process all images
    python calculate_stats.py --max-samples 1000 # Process first 1000 images
    python calculate_stats.py --dataset-root /path/to/dataset
"""

import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from dataset import load_dataset_paths


def calculate_dataset_stats(dataset_root="../Dataset", max_samples=None):
    """
    Calculate mean and standard deviation across training images.

    Args:
        dataset_root: Path to dataset directory
        max_samples: Maximum number of images to process (None for all)

    Returns:
        tuple: (mean, std) as numpy arrays of shape (3,) for RGB channels
    """
    print("=" * 60)
    print("Calculating Dataset Statistics")
    print("=" * 60)

    # Load dataset paths
    print(f"\n📁 Loading dataset from: {dataset_root}")
    try:
        datasets = load_dataset_paths(dataset_root)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None, None

    train_paths, _ = datasets['train']

    # Limit samples if specified
    if max_samples is not None and max_samples < len(train_paths):
        train_paths = train_paths[:max_samples]
        print(f"📊 Processing {max_samples} images (subset of {len(datasets['train'][0])} total)")
    else:
        print(f"📊 Processing all {len(train_paths)} training images")

    # Initialize accumulators
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sum_sq = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    print("\n🔄 Processing images...")

    # Process each image
    for img_path in tqdm(train_paths, desc="Processing images"):
        try:
            # Read image
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"⚠️  Warning: Could not read {img_path}")
                continue

            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Convert to float and normalize to [0, 1]
            img_float = img_rgb.astype(np.float64) / 255.0

            # Accumulate statistics
            pixel_sum += np.sum(img_float, axis=(0, 1))
            pixel_sum_sq += np.sum(img_float ** 2, axis=(0, 1))
            total_pixels += img_float.shape[0] * img_float.shape[1]

        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
            continue

    if total_pixels == 0:
        print("❌ No valid images found!")
        return None, None

    # Calculate mean and std
    mean = pixel_sum / total_pixels
    variance = (pixel_sum_sq / total_pixels) - (mean ** 2)
    std = np.sqrt(np.maximum(variance, 0))  # Ensure non-negative

    print("\n✅ Statistics calculated successfully!")
    print(f"📈 Total pixels processed: {total_pixels:,}")
    print(f"📊 Mean (RGB): [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]")
    print(f"📊 Std  (RGB): [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")

    print("\n💡 Use these values in your config:")
    print(f"   'normalize': {{")
    print(f"       'mean': ({mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}),")
    print(f"       'std': ({std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f})")
    print(f"   }}")
    print(f"\n📝 Note: Current config uses ImageNet normalization.")
    print(f"   Dataset-specific normalization may improve performance.")

    return mean, std


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate dataset statistics for normalization")
    parser.add_argument("--dataset-root", default="../Dataset",
                       help="Path to dataset directory")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of images to process (default: all)")

    args = parser.parse_args()

    # Calculate stats for the dataset
    mean, std = calculate_dataset_stats(args.dataset_root, args.max_samples)

    if mean is not None and std is not None:
        print("\n" + "=" * 60)
        print("Dataset statistics calculation completed!")
        print("=" * 60)
    else:
        print("\n❌ Failed to calculate dataset statistics!")
        exit(1)
