#!/usr/bin/env python3
"""
Restructure ddata directory to match the format expected by DeepfakeDataModule:
- Train/ (capital T)
  - 0_real/
  - 1_fake/
- Val/ (capital V)
  - 0_real/
  - 1_fake/
- Test/ (capital T)
  - 0_real/
  - 1_fake/
"""

import os
import shutil
from pathlib import Path

def restructure_ddata(source_dir):
    """
    Restructure ddata directory to match DeepfakeDataModule expectations.
    """
    print("="*60)
    print("RESTRUCTURING DDATA DIRECTORY")
    print("="*60)
    
    source_path = Path(source_dir)
    
    # Mapping: old structure -> new structure
    # Split directories: lowercase -> capitalized
    # Class directories: fake/real -> 1_fake/0_real
    
    splits = [
        ('train', 'Train'),
        ('val', 'Val'),
        ('test', 'Test')
    ]
    
    class_mapping = [
        ('fake', '1_fake'),
        ('real', '0_real')
    ]
    
    print(f"\n📂 Source directory: {source_path}")
    
    # Process each split
    for old_split, new_split in splits:
        old_split_dir = source_path / old_split
        new_split_dir = source_path / new_split
        
        if not old_split_dir.exists():
            print(f"  ⚠️  Warning: {old_split_dir} does not exist, skipping...")
            continue
        
        print(f"\n📦 Processing {old_split} -> {new_split}...")
        
        # Create new split directory
        new_split_dir.mkdir(exist_ok=True)
        
        # Process each class
        for old_class, new_class in class_mapping:
            old_class_dir = old_split_dir / old_class
            new_class_dir = new_split_dir / new_class
            
            if not old_class_dir.exists():
                print(f"  ⚠️  Warning: {old_class_dir} does not exist, skipping...")
                continue
            
            # Count files before move
            num_files = len(list(old_class_dir.iterdir()))
            
            # Move the entire directory
            print(f"    Moving {old_class}/ -> {new_class}/ ({num_files} files)...")
            
            # If destination exists, merge by moving files
            if new_class_dir.exists():
                print(f"      Destination exists, merging files...")
                for file_path in old_class_dir.iterdir():
                    if file_path.is_file():
                        shutil.move(str(file_path), str(new_class_dir / file_path.name))
                # Remove empty old directory
                try:
                    old_class_dir.rmdir()
                except OSError:
                    pass
            else:
                # Rename the directory
                shutil.move(str(old_class_dir), str(new_class_dir))
            
            # Verify
            if new_class_dir.exists():
                actual_count = len(list(new_class_dir.iterdir()))
                print(f"      ✓ {new_class}: {actual_count} files")
        
        # Remove old split directory if empty
        try:
            old_split_dir.rmdir()
            print(f"  ✓ Removed empty {old_split}/ directory")
        except OSError:
            # Directory not empty or doesn't exist
            pass
    
    # Final verification
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}\n")
    
    expected_splits = ['Train', 'Val', 'Test']
    expected_classes = ['0_real', '1_fake']
    
    for split in expected_splits:
        split_dir = source_path / split
        if split_dir.exists():
            print(f"✅ {split}/ exists")
            for class_name in expected_classes:
                class_dir = split_dir / class_name
                if class_dir.exists():
                    num_files = len(list(class_dir.iterdir()))
                    print(f"   ✅ {class_name}/: {num_files} files")
                else:
                    print(f"   ❌ {class_name}/: MISSING")
        else:
            print(f"❌ {split}/: MISSING")
    
    print(f"\n{'='*60}")
    print("✅ Restructuring complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    DDATA_DIR = "/Users/sameerkashyap/code/ctd-fusionnet-deepfake-detection/ddata"
    restructure_ddata(DDATA_DIR)

