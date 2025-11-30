#!/usr/bin/env python3
"""
Check for dataset issues that could cause 100% validation accuracy:
- Data leakage between splits
- Label issues
- Class imbalance
"""

import os
import sys

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

def main():
    dataset_root = CONFIG['dataset_root']
    
    if not os.path.exists(dataset_root):
        print(f"❌ Dataset root not found: {dataset_root}")
        return
    
    train_dir, val_dir, test_dir = find_dataset_dirs(dataset_root)
    
    if not all([train_dir, val_dir, test_dir]):
        print(f"❌ Missing directories. Train: {train_dir}, Val: {val_dir}, Test: {test_dir}")
        return
    
    # Load paths
    train_real = list_images(os.path.join(train_dir, 'Real'))
    train_fake = list_images(os.path.join(train_dir, 'Fake'))
    val_real = list_images(os.path.join(val_dir, 'Real'))
    val_fake = list_images(os.path.join(val_dir, 'Fake'))
    test_real = list_images(os.path.join(test_dir, 'Real'))
    test_fake = list_images(os.path.join(test_dir, 'Fake'))
    
    train_paths = train_real + train_fake
    train_labels = [0] * len(train_real) + [1] * len(train_fake)
    val_paths = val_real + val_fake
    val_labels = [0] * len(val_real) + [1] * len(val_fake)
    test_paths = test_real + test_fake
    test_labels = [0] * len(test_real) + [1] * len(test_fake)
    
    print("=" * 60)
    print("DATASET DIAGNOSTICS")
    print("=" * 60)
    
    # Dataset sizes
    print(f"\n📊 Dataset Sizes:")
    print(f"  Train: {len(train_paths)} images ({len(train_real)} real, {len(train_fake)} fake)")
    print(f"  Val:   {len(val_paths)} images ({len(val_real)} real, {len(val_fake)} fake)")
    print(f"  Test:  {len(test_paths)} images ({len(test_real)} real, {len(test_fake)} fake)")
    
    # Check for filename overlap (data leakage)
    print(f"\n🔍 Checking for Data Leakage (Filename Overlap):")
    train_names = set(os.path.basename(p) for p in train_paths)
    val_names = set(os.path.basename(p) for p in val_paths)
    test_names = set(os.path.basename(p) for p in test_paths)
    
    train_val_overlap = train_names & val_names
    train_test_overlap = train_names & test_names
    val_test_overlap = val_names & test_names
    
    print(f"  Train ↔ Val:   {len(train_val_overlap)} duplicates")
    if train_val_overlap:
        print(f"    ⚠️  CRITICAL: Found duplicates! Examples: {list(train_val_overlap)[:5]}")
    else:
        print(f"    ✅ No duplicates")
        
    print(f"  Train ↔ Test:  {len(train_test_overlap)} duplicates")
    if train_test_overlap:
        print(f"    ⚠️  CRITICAL: Found duplicates! Examples: {list(train_test_overlap)[:5]}")
    else:
        print(f"    ✅ No duplicates")
        
    print(f"  Val ↔ Test:    {len(val_test_overlap)} duplicates")
    if val_test_overlap:
        print(f"    ⚠️  CRITICAL: Found duplicates! Examples: {list(val_test_overlap)[:5]}")
    else:
        print(f"    ✅ No duplicates")
    
    # Check label distribution
    print(f"\n📈 Label Distribution:")
    print(f"  Train: {sum(train_labels)}/{len(train_labels)} fake ({100*sum(train_labels)/len(train_labels):.1f}%)")
    print(f"  Val:   {sum(val_labels)}/{len(val_labels)} fake ({100*sum(val_labels)/len(val_labels):.1f}%)")
    print(f"  Test:  {sum(test_labels)}/{len(test_labels)} fake ({100*sum(test_labels)/len(test_labels):.1f}%)")
    
    # Check if validation set is all one class
    if len(set(val_labels)) == 1:
        print(f"\n❌ CRITICAL: Validation set has only ONE class!")
        print(f"    All validation samples are: {'Fake' if val_labels[0] == 1 else 'Real'}")
        print(f"    This would lead to 100% accuracy if model predicts only that class!")
    elif sum(val_labels) == 0 or sum(val_labels) == len(val_labels):
        print(f"\n⚠️  WARNING: Validation set is highly imbalanced!")
        print(f"    This could cause misleading metrics.")
    else:
        print(f"  ✅ Validation set has both classes")
    
    # Check for video ID overlap (more subtle leakage)
    print(f"\n🔍 Checking for Video ID Overlap (Potential Subtle Leakage):")
    
    def extract_video_id(filename):
        """Extract video ID from filename (heuristic)."""
        # Common patterns: videoID_frameXXX.jpg or just frameXXX.jpg
        parts = filename.split('_')
        if len(parts) >= 2:
            return '_'.join(parts[:2])  # Take first two parts
        elif len(parts) == 1:
            # Try to extract numeric prefix
            base = os.path.splitext(parts[0])[0]
            # If it starts with a word, use that
            if base[0].isalpha():
                i = 0
                while i < len(base) and (base[i].isalnum() or base[i] == '_'):
                    i += 1
                return base[:i]
            return base
        return filename
    
    train_video_ids = set(extract_video_id(os.path.basename(p)) for p in train_paths)
    val_video_ids = set(extract_video_id(os.path.basename(p)) for p in val_paths)
    test_video_ids = set(extract_video_id(os.path.basename(p)) for p in test_paths)
    
    train_val_video_overlap = train_video_ids & val_video_ids
    train_test_video_overlap = train_video_ids & test_video_ids
    
    print(f"  Unique video IDs - Train: {len(train_video_ids)}, Val: {len(val_video_ids)}, Test: {len(test_video_ids)}")
    print(f"  Train ↔ Val video overlap: {len(train_val_video_overlap)}")
    if train_val_video_overlap:
        print(f"    ⚠️  WARNING: Same videos appear in both Train and Val!")
        print(f"    Examples: {list(train_val_video_overlap)[:5]}")
        print(f"    This is subtle data leakage - frames from same video in both sets!")
    else:
        print(f"    ✅ No video ID overlap")
    
    print(f"  Train ↔ Test video overlap: {len(train_test_video_overlap)}")
    if train_test_video_overlap:
        print(f"    ⚠️  WARNING: Same videos appear in both Train and Test!")
        print(f"    Examples: {list(train_test_video_overlap)[:5]}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("SUMMARY:")
    issues = []
    if train_val_overlap or train_test_overlap:
        issues.append("Exact filename duplicates found")
    if train_val_video_overlap:
        issues.append("Video ID overlap between Train and Val (subtle leakage)")
    if len(set(val_labels)) == 1:
        issues.append("Validation set has only one class")
    if sum(val_labels) == 0 or sum(val_labels) == len(val_labels):
        issues.append("Validation set is completely imbalanced")
    
    if issues:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ No obvious issues found. 100% accuracy might be legitimate or due to:")
        print("  - Easy dataset (images very different)")
        print("  - Overfitting (model memorized training data)")
        print("  - Hidden metadata in filenames/paths")
        print("  - Temporal/spatial patterns in data organization")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

