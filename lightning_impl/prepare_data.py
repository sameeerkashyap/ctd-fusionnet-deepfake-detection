import kagglehub
from pathlib import Path
from sklearn.model_selection import train_test_split
import os
import shutil

def prepare_data():
    # 1. Download the dataset
    print("Downloading dataset from Kaggle...")
    # Dataset URL : https://www.kaggle.com/datasets/ayushmandatta1/deepdetect-2025
    path = kagglehub.dataset_download("ayushmandatta1/deepdetect-2025")
    print("Path to dataset files:", path)
    
    # Define paths
    # The downloaded dataset path usually contains the 'ddata' folder inside
    ddata = Path(path) / "ddata"
    
    # We'll create a working copy in ./data (read-write space)
    work_data = Path("./data")
    
    # Check if data already exists to avoid re-doing work if not needed, 
    # but for this script we will assume we want to ensure the structure is correct.
    # If you want to force re-creation, you might want to clean up ./data first.
    if work_data.exists():
        print(f"Directory {work_data} already exists. Cleaning up to ensure fresh split...")
        shutil.rmtree(work_data)
    
    work_data.mkdir(parents=True, exist_ok=True)

    # 2. Create folder structure
    print("Creating folder structure...")
    for split in ("train", "val", "test"):
        for cls in ("fake", "real"):
            (work_data / split / cls).mkdir(parents=True, exist_ok=True)

    # 3. Collect original train images
    print("Collecting image paths from original train folder...")
    train_images = {"fake": [], "real": []}

    for cls in ("fake", "real"):
        folder = ddata / "train" / cls
        # Collect common image extensions
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + \
                 list(folder.glob("*.png")) + list(folder.glob("*.webp"))
        train_images[cls] = images

    print(f"Original train set:")
    print(f"  - fake: {len(train_images['fake']):,}")
    print(f"  - real: {len(train_images['real']):,}")
    print(f"  - total: {len(train_images['fake']) + len(train_images['real']):,}")

    # 4. Split each class: 90% train, 10% val (stratified)
    train_split = {"fake": [], "real": []}
    val_split = {"fake": [], "real": []}

    for cls in ("fake", "real"):
        # Only split if we have images
        if len(train_images[cls]) > 0:
            train_cls, val_cls = train_test_split(
                train_images[cls], 
                test_size=0.1,      # 10% for validation
                random_state=42,    # Reproducibility
                shuffle=True
            )
            train_split[cls] = train_cls
            val_split[cls] = val_cls
        else:
            print(f"Warning: No images found for class {cls} in training set.")

    print(f"\nAfter 90/10 split:")
    print(f"Train set:")
    print(f"  - fake: {len(train_split['fake']):,}")
    print(f"  - real: {len(train_split['real']):,}")
    print(f"  - total: {len(train_split['fake']) + len(train_split['real']):,}")
    print(f"\nValidation set:")
    print(f"  - fake: {len(val_split['fake']):,}")
    print(f"  - real: {len(val_split['real']):,}")
    print(f"  - total: {len(val_split['fake']) + len(val_split['real']):,}")

    # 5. Create symlinks (faster than copying)
    print("\nCreating symlinks to new train/val folders...")

    for cls in ("fake", "real"):
        # Train links
        for img_path in train_split[cls]:
            link = work_data / "train" / cls / img_path.name
            if not link.exists():
                os.symlink(img_path, link)
        
        # Val links
        for img_path in val_split[cls]:
            link = work_data / "val" / cls / img_path.name
            if not link.exists():
                os.symlink(img_path, link)

    # 6. Link test set for completeness
    print("Linking test set...")
    for cls in ("fake", "real"):
        test_folder = ddata / "test" / cls
        if test_folder.exists():
            for img_path in test_folder.iterdir():
                if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
                    link = work_data / "test" / cls / img_path.name
                    if not link.exists():
                        os.symlink(img_path, link)
        else:
             print(f"Warning: Test folder for {cls} not found at {test_folder}")

    # Verify final counts
    print(f"\n{'='*60}")
    print("✅ FINAL DATASET STRUCTURE")
    print(f"{'='*60}")
    print(f"Location: {work_data.resolve()}")

    for split in ("train", "val", "test"):
        print(f"\n{split.capitalize()}:")
        for cls in ("fake", "real"):
            count = len(list((work_data / split / cls).iterdir()))
            print(f"  - {cls}: {count:,}")

if __name__ == "__main__":
    prepare_data()
