#!/usr/bin/env python3
"""
Split CatandDog dataset into Train/Val/Test structure compatible with debug_dataset.py

- Takes 10% of training data (balanced across cats and dogs)
- Splits into Train (90%) and Val (10%) from the 10% subset
- Uses existing test_set for Test
- Maps: cats -> Real (0), dogs -> Fake (1)
"""

import os
import shutil
import random
from pathlib import Path

def get_image_files(directory):
    """Get all .jpg image files from a directory."""
    return sorted([f for f in os.listdir(directory) if f.lower().endswith('.jpg')])

def copy_files(src_files, src_dir, dst_dir, num_files):
    """Copy specified number of files from src_dir to dst_dir."""
    os.makedirs(dst_dir, exist_ok=True)
    for filename in src_files[:num_files]:
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        shutil.copy2(src_path, dst_path)

def main():
    # Paths
    base_dir = Path("/Users/sameerkashyap/code/ctd-fusionnet-deepfake-detection/CatandDog")
    output_dir = base_dir / "split_dataset"
    
    training_cats_dir = base_dir / "training_set" / "training_set" / "cats"
    training_dogs_dir = base_dir / "training_set" / "training_set" / "dogs"
    test_cats_dir = base_dir / "test_set" / "test_set" / "cats"
    test_dogs_dir = base_dir / "test_set" / "test_set" / "dogs"
    
    # Output directories
    train_real_dir = output_dir / "Train" / "Real"
    train_fake_dir = output_dir / "Train" / "Fake"
    val_real_dir = output_dir / "Val" / "Real"
    val_fake_dir = output_dir / "Val" / "Fake"
    test_real_dir = output_dir / "Test" / "Real"
    test_fake_dir = output_dir / "Test" / "Fake"
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Get all training images
    print("Loading training images...")
    cat_images = get_image_files(training_cats_dir)
    dog_images = get_image_files(training_dogs_dir)
    
    print(f"Total training images: {len(cat_images)} cats, {len(dog_images)} dogs")
    
    # Take 10% of each class
    num_cats_10pct = max(1, int(len(cat_images) * 0.10))
    num_dogs_10pct = max(1, int(len(dog_images) * 0.10))
    
    print(f"Taking 10%: {num_cats_10pct} cats, {num_dogs_10pct} dogs")
    
    # Shuffle and take 10%
    random.shuffle(cat_images)
    random.shuffle(dog_images)
    
    selected_cats = cat_images[:num_cats_10pct]
    selected_dogs = dog_images[:num_dogs_10pct]
    
    # Split 10% subset into Train (90%) and Val (10%)
    num_cats_train = int(num_cats_10pct * 0.9)
    num_dogs_train = int(num_dogs_10pct * 0.9)
    
    num_cats_val = num_cats_10pct - num_cats_train
    num_dogs_val = num_dogs_10pct - num_dogs_train
    
    print(f"\nSplitting into Train/Val:")
    print(f"  Cats - Train: {num_cats_train}, Val: {num_cats_val}")
    print(f"  Dogs - Train: {num_dogs_train}, Val: {num_dogs_val}")
    
    # Copy training images (cats -> Real, dogs -> Fake)
    print("\nCopying Train images...")
    copy_files(selected_cats[:num_cats_train], training_cats_dir, train_real_dir, num_cats_train)
    copy_files(selected_dogs[:num_dogs_train], training_dogs_dir, train_fake_dir, num_dogs_train)
    
    # Copy validation images
    print("Copying Val images...")
    copy_files(selected_cats[num_cats_train:], training_cats_dir, val_real_dir, num_cats_val)
    copy_files(selected_dogs[num_dogs_train:], training_dogs_dir, val_fake_dir, num_dogs_val)
    
    # Copy test images
    print("Copying Test images...")
    test_cat_images = get_image_files(test_cats_dir)
    test_dog_images = get_image_files(test_dogs_dir)
    print(f"Test images: {len(test_cat_images)} cats, {len(test_dog_images)} dogs")
    
    copy_files(test_cat_images, test_cats_dir, test_real_dir, len(test_cat_images))
    copy_files(test_dog_images, test_dogs_dir, test_fake_dir, len(test_dog_images))
    
    # Summary
    print("\n" + "="*50)
    print("Dataset split complete!")
    print("="*50)
    print(f"Output directory: {output_dir}")
    print(f"\nTrain:")
    print(f"  Real (cats): {len(os.listdir(train_real_dir))} images")
    print(f"  Fake (dogs): {len(os.listdir(train_fake_dir))} images")
    print(f"\nVal:")
    print(f"  Real (cats): {len(os.listdir(val_real_dir))} images")
    print(f"  Fake (dogs): {len(os.listdir(val_fake_dir))} images")
    print(f"\nTest:")
    print(f"  Real (cats): {len(os.listdir(test_real_dir))} images")
    print(f"  Fake (dogs): {len(os.listdir(test_fake_dir))} images")
    print("\nTo use with debug_dataset.py, set dataset_root to:")
    print(f"  {output_dir}")

if __name__ == "__main__":
    main()

