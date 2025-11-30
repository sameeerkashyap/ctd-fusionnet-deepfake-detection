#!/usr/bin/env python3
"""
Deep diagnostic for 100% validation accuracy issue.
Checks for:
- Filename/path patterns that leak information
- Image statistics differences
- Model predictions breakdown
"""

import os
import sys
import numpy as np
from collections import Counter

CONFIG = {
    'dataset_root': "/Users/sameerkashyap/code/ctd-fusionnet-deepfake-detection/Diffusion",
}

def list_images(folder):
    """List all image files in a folder."""
    if not os.path.exists(folder):
        return []
    return sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    ])

def find_dataset_dirs(base_dir):
    """Find Train, Val, and Test directories."""
    train_dir = None
    val_dir = None
    test_dir = None
    
    for root, dirs, _ in os.walk(base_dir):
        if 'Train' in dirs:
            train_dir = os.path.join(root, 'Train')
        if 'Val' in dirs:
            val_dir = os.path.join(root, 'Val')
        if 'Test' in dirs:
            test_dir = os.path.join(root, 'Test')
    
    return train_dir, val_dir, test_dir

def analyze_filename_patterns(paths, labels, name):
    """Check if filenames contain patterns that leak label information."""
    print(f"\n🔍 Analyzing Filename Patterns for {name}:")
    
    real_paths = [p for p, l in zip(paths, labels) if l == 0]
    fake_paths = [p for p, l in zip(paths, labels) if l == 1]
    
    real_filenames = [os.path.basename(p) for p in real_paths[:100]]
    fake_filenames = [os.path.basename(p) for p in fake_paths[:100]]
    
    # Check for common prefixes
    real_prefixes = [f.split('_')[0] for f in real_filenames]
    fake_prefixes = [f.split('_')[0] for f in fake_filenames]
    
    real_prefix_counts = Counter(real_prefixes)
    fake_prefix_counts = Counter(fake_prefixes)
    
    print(f"  Top Real prefixes: {real_prefix_counts.most_common(5)}")
    print(f"  Top Fake prefixes: {fake_prefix_counts.most_common(5)}")
    
    # Check if any prefix is exclusively real or fake
    all_real_prefixes = set(real_prefix_counts.keys())
    all_fake_prefixes = set(fake_prefix_counts.keys())
    
    exclusive_real = all_real_prefixes - all_fake_prefixes
    exclusive_fake = all_fake_prefixes - all_real_prefixes
    
    if exclusive_real:
        print(f"  ⚠️  Prefixes ONLY in Real: {list(exclusive_real)[:10]}")
    if exclusive_fake:
        print(f"  ⚠️  Prefixes ONLY in Fake: {list(exclusive_fake)[:10]}")
    
    # Check directory structure
    real_dirs = set(os.path.dirname(p) for p in real_paths[:100])
    fake_dirs = set(os.path.dirname(p) for p in fake_paths[:100])
    
    print(f"  Real dirs: {len(real_dirs)} unique directories")
    print(f"  Fake dirs: {len(fake_dirs)} unique directories")
    
    # Check if path contains "real" or "fake"
    real_paths_lower = [p.lower() for p in real_paths[:100]]
    fake_paths_lower = [p.lower() for p in fake_paths[:100]]
    
    real_has_label_in_path = sum(1 for p in real_paths_lower if 'real' in p)
    fake_has_label_in_path = sum(1 for p in fake_paths_lower if 'fake' in p)
    
    print(f"  Paths containing 'real': {real_has_label_in_path}/{len(real_paths_lower)}")
    print(f"  Paths containing 'fake': {fake_has_label_in_path}/{len(fake_paths_lower)}")
    
    if real_has_label_in_path == len(real_paths_lower) and fake_has_label_in_path == len(fake_paths_lower):
        print(f"  ⚠️  WARNING: All paths contain label information!")
        print(f"     The model might be learning from file paths, not images!")

def analyze_image_file_sizes(paths, labels, name, sample_size=200):
    """Check if file sizes differ significantly between real and fake."""
    print(f"\n📦 Analyzing File Sizes for {name}:")
    
    real_paths = [p for p, l in zip(paths, labels) if l == 0]
    fake_paths = [p for p, l in zip(paths, labels) if l == 1]
    
    real_sizes = []
    fake_sizes = []
    
    for p in real_paths[:sample_size]:
        if os.path.exists(p):
            real_sizes.append(os.path.getsize(p))
    
    for p in fake_paths[:sample_size]:
        if os.path.exists(p):
            fake_sizes.append(os.path.getsize(p))
    
    if real_sizes and fake_sizes:
        real_mean = np.mean(real_sizes)
        fake_mean = np.mean(fake_sizes)
        real_std = np.std(real_sizes)
        fake_std = np.std(fake_sizes)
        
        print(f"  Real: mean={real_mean:.0f} bytes, std={real_std:.0f}")
        print(f"  Fake: mean={fake_mean:.0f} bytes, std={fake_std:.0f}")
        
        diff = abs(real_mean - fake_mean)
        if diff > 10000:
            print(f"  ⚠️  Large difference in file sizes ({diff:.0f} bytes)")
            print(f"     This could be a feature the model learns!")

def main():
    dataset_root = CONFIG['dataset_root']
    
    if not os.path.exists(dataset_root):
        print(f"❌ Dataset root not found: {dataset_root}")
        return
    
    train_dir, val_dir, test_dir = find_dataset_dirs(dataset_root)
    
    # Load paths
    train_real = list_images(os.path.join(train_dir, 'Real'))
    train_fake = list_images(os.path.join(train_dir, 'Fake'))
    val_real = list_images(os.path.join(val_dir, 'Real'))
    val_fake = list_images(os.path.join(val_dir, 'Fake'))
    
    train_paths = train_real + train_fake
    train_labels = [0] * len(train_real) + [1] * len(train_fake)
    val_paths = val_real + val_fake
    val_labels = [0] * len(val_real) + [1] * len(val_fake)
    
    print("=" * 70)
    print("DEEP DIAGNOSTIC: Why 100% Validation Accuracy?")
    print("=" * 70)
    
    # Check filename patterns
    analyze_filename_patterns(train_paths, train_labels, "Train")
    analyze_filename_patterns(val_paths, val_labels, "Validation")
    
    # Check file sizes
    analyze_image_file_sizes(train_paths, train_labels, "Train")
    analyze_image_file_sizes(val_paths, val_labels, "Validation")
    
    # Check if validation set has any obvious pattern
    print(f"\n🔍 Validation Set Structure:")
    print(f"  Sample Real files: {[os.path.basename(p) for p in val_real[:5]]}")
    print(f"  Sample Fake files: {[os.path.basename(p) for p in val_fake[:5]]}")
    
    # Check for perfect separation in filenames
    val_real_names = set(os.path.basename(p).lower() for p in val_real)
    val_fake_names = set(os.path.basename(p).lower() for p in val_fake)
    
    # Check if there's a pattern that perfectly separates them
    # E.g., all real start with 'r' or all fake have certain characters
    if val_real_names and val_fake_names:
        # Check first character
        real_first_chars = set(f[0] for f in val_real_names if f)
        fake_first_chars = set(f[0] for f in val_fake_names if f)
        
        if real_first_chars.isdisjoint(fake_first_chars):
            print(f"  ⚠️  CRITICAL: Real and Fake files have DIFFERENT first characters!")
            print(f"     Real starts with: {sorted(real_first_chars)}")
            print(f"     Fake starts with: {sorted(fake_first_chars)}")
            print(f"     Model might be learning from filenames!")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("1. Ensure images are loaded correctly (not paths)")
    print("2. Check if model is somehow accessing filenames")
    print("3. Verify image preprocessing is consistent")
    print("4. Add debugging to see actual model predictions vs labels")
    print("=" * 70)

if __name__ == "__main__":
    main()

