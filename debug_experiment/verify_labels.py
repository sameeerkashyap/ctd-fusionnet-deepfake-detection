#!/usr/bin/env python3
"""
Quick script to verify labels are being created correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import load_dataset_paths

# Load dataset paths
dataset_root = "/Users/sameerkashyap/code/ctd-fusionnet-deepfake-detection/Diffusion"
datasets = load_dataset_paths(dataset_root)

train_paths, train_labels = datasets['train']
val_paths, val_labels = datasets['val']
test_paths, test_labels = datasets['test']

print("=" * 60)
print("LABEL VERIFICATION")
print("=" * 60)

# Check label values
print(f"\nTrain labels:")
print(f"  Total: {len(train_labels)}")
print(f"  Unique values: {set(train_labels)}")
print(f"  Label distribution: {train_labels.count(0)} real, {train_labels.count(1)} fake")
print(f"  Sample labels (first 10): {train_labels[:10]}")

print(f"\nVal labels:")
print(f"  Total: {len(val_labels)}")
print(f"  Unique values: {set(val_labels)}")
print(f"  Label distribution: {val_labels.count(0)} real, {val_labels.count(1)} fake")
print(f"  Sample labels (first 10): {val_labels[:10]}")

print(f"\nTest labels:")
print(f"  Total: {len(test_labels)}")
print(f"  Unique values: {set(test_labels)}")
print(f"  Label distribution: {test_labels.count(0)} real, {test_labels.count(1)} fake")
print(f"  Sample labels (first 10): {test_labels[:10]}")

# Verify labels match paths
print(f"\nVerifying label-path correspondence:")
print(f"  Train: {train_paths[0]} -> label {train_labels[0]}")
print(f"  Train: {train_paths[-1]} -> label {train_labels[-1]}")

# Check if all labels are 0 or 1
all_labels = train_labels + val_labels + test_labels
if all(label in [0, 1] for label in all_labels):
    print(f"\n✅ All labels are valid (0 or 1)")
else:
    invalid = [l for l in all_labels if l not in [0, 1]]
    print(f"\n❌ Found invalid labels: {set(invalid)}")

print("=" * 60)

