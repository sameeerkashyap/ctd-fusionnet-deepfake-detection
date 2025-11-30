#!/usr/bin/env python3
"""
Split ddata/train into train and val sets with stratification.
Preserves the class distribution (fake/real) in both splits.
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def get_all_images(directory):
    """Get all image files (.jpg, .png, .jpeg) from a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    files = []
    if not os.path.exists(directory):
        return files
    for f in os.listdir(directory):
        if any(f.endswith(ext) for ext in image_extensions):
            files.append(f)
    return sorted(files)

def split_train_val_stratified(source_train_dir, val_ratio=0.2, random_seed=42):
    """
    Split train directory into train and val with stratification.
    
    Args:
        source_train_dir: Path to the train directory (e.g., ddata/train)
        val_ratio: Proportion of data to use for validation (default: 0.2 = 20%)
        random_seed: Random seed for reproducibility
    """
    print("="*60)
    print("SPLITTING TRAIN INTO TRAIN AND VAL WITH STRATIFICATION")
    print("="*60)
    
    # Set random seed
    random.seed(random_seed)
    
    # Get base directory
    base_dir = Path(source_train_dir).parent
    train_dir = Path(source_train_dir)
    val_dir = base_dir / "val"
    
    # Class directories
    fake_train_dir = train_dir / "fake"
    real_train_dir = train_dir / "real"
    
    # Create validation directories
    fake_val_dir = val_dir / "fake"
    real_val_dir = val_dir / "real"
    
    # Check if source directories exist
    if not fake_train_dir.exists():
        raise FileNotFoundError(f"Fake directory not found: {fake_train_dir}")
    if not real_train_dir.exists():
        raise FileNotFoundError(f"Real directory not found: {real_train_dir}")
    
    # Get all image files for each class
    print(f"\n📂 Scanning source directories...")
    fake_images = get_all_images(fake_train_dir)
    real_images = get_all_images(real_train_dir)
    
    print(f"  ✓ Found {len(fake_images)} fake images")
    print(f"  ✓ Found {len(real_images)} real images")
    print(f"  ✓ Total: {len(fake_images) + len(real_images)} images")
    
    # Prepare data for stratified split
    # We need to create labels and file paths
    all_files = []
    all_labels = []
    
    for img in fake_images:
        all_files.append((fake_train_dir, img, "fake"))
        all_labels.append("fake")
    
    for img in real_images:
        all_files.append((real_train_dir, img, "real"))
        all_labels.append("real")
    
    # Manual stratified split to maintain class distribution
    print(f"\n🔄 Performing stratified split ({val_ratio*100:.1f}% validation)...")
    
    # Group files by class
    fake_files = [f for f in all_files if f[2] == "fake"]
    real_files = [f for f in all_files if f[2] == "real"]
    
    # Shuffle each class separately
    random.shuffle(fake_files)
    random.shuffle(real_files)
    
    # Calculate split sizes for each class
    num_fake_val = int(len(fake_files) * val_ratio)
    num_real_val = int(len(real_files) * val_ratio)
    
    # Split each class
    fake_train = fake_files[num_fake_val:]
    fake_val = fake_files[:num_fake_val]
    real_train = real_files[num_real_val:]
    real_val = real_files[:num_real_val]
    
    # Combine splits
    train_files = fake_train + real_train
    val_files = fake_val + real_val
    
    # Shuffle the combined lists
    random.shuffle(train_files)
    random.shuffle(val_files)
    
    print(f"  ✓ Train: {len(train_files)} images ({len(fake_train)} fake, {len(real_train)} real)")
    print(f"  ✓ Val: {len(val_files)} images ({len(fake_val)} fake, {len(real_val)} real)")
    
    # Create validation directories
    print(f"\n📁 Creating validation directories...")
    fake_val_dir.mkdir(parents=True, exist_ok=True)
    real_val_dir.mkdir(parents=True, exist_ok=True)
    
    # Move validation files to val directory
    print(f"\n📦 Moving validation files...")
    val_fake_count = 0
    val_real_count = 0
    
    for src_dir, filename, label in val_files:
        src_path = src_dir / filename
        if label == "fake":
            dst_path = fake_val_dir / filename
            val_fake_count += 1
        else:
            dst_path = real_val_dir / filename
            val_real_count += 1
        
        shutil.move(str(src_path), str(dst_path))
    
    print(f"  ✓ Moved {val_fake_count} fake images to {fake_val_dir}")
    print(f"  ✓ Moved {val_real_count} real images to {real_val_dir}")
    
    # Verify remaining train files
    print(f"\n✅ Split complete!")
    print(f"\n📊 Final distribution:")
    
    remaining_fake = len(get_all_images(fake_train_dir))
    remaining_real = len(get_all_images(real_train_dir))
    val_fake = len(get_all_images(fake_val_dir))
    val_real = len(get_all_images(real_val_dir))
    
    print(f"\n  Train:")
    print(f"    Fake: {remaining_fake} images")
    print(f"    Real: {remaining_real} images")
    print(f"    Total: {remaining_fake + remaining_real} images")
    
    print(f"\n  Val:")
    print(f"    Fake: {val_fake} images")
    print(f"    Real: {val_real} images")
    print(f"    Total: {val_fake + val_real} images")
    
    # Calculate proportions
    train_total = remaining_fake + remaining_real
    val_total = val_fake + val_real
    total = train_total + val_total
    
    if total > 0:
        print(f"\n  Class distribution:")
        print(f"    Fake: {(remaining_fake + val_fake) / total * 100:.1f}% total ({remaining_fake} train, {val_fake} val)")
        print(f"    Real: {(remaining_real + val_real) / total * 100:.1f}% total ({remaining_real} train, {val_real} val)")
    
    print(f"\n{'='*60}")
    print(f"✅ Stratified split complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Configuration
    DDATA_TRAIN_DIR = "/Users/sameerkashyap/code/ctd-fusionnet-deepfake-detection/ddata/train"
    VAL_RATIO = 0.2  # 20% for validation, 80% for training
    RANDOM_SEED = 42  # For reproducibility
    
    split_train_val_stratified(DDATA_TRAIN_DIR, val_ratio=VAL_RATIO, random_seed=RANDOM_SEED)

