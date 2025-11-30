
import os
import sys
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import load_dataset_paths

CONFIG = {
    'dataset_root': "../Dataset",
}

def check_leakage(train_paths, test_paths):
    print("\n🔍 Checking for Data Leakage (Filename Overlap)...")
    
    # Extract filenames
    train_names = set([os.path.basename(p) for p in train_paths])
    test_names = set([os.path.basename(p) for p in test_paths])
    
    # Check exact duplicates
    intersection = train_names.intersection(test_names)
    if len(intersection) > 0:
        print(f"❌ CRITICAL: Found {len(intersection)} exact filename duplicates in Train and Test!")
        print(f"   Examples: {list(intersection)[:5]}")
    else:
        print("✅ No exact filename duplicates found.")
        
    # Check Video ID overlap (assuming format video_id_frame.jpg)
    # Heuristic: Split by underscore and take first 2 parts as ID (common in DF datasets)
    train_ids = set(["_".join(f.split('_')[:2]) for f in train_names])
    test_ids = set(["_".join(f.split('_')[:2]) for f in test_names])
    
    id_intersection = train_ids.intersection(test_ids)
    if len(id_intersection) > 0:
        print(f"⚠️  WARNING: Found {len(id_intersection)} potential Video ID overlaps!")
        print(f"   Examples: {list(id_intersection)[:5]}")
    else:
        print("✅ No Video ID overlaps found (based on heuristic).")

def check_image_stats(paths, label_name):
    print(f"\n📊 Checking Image Statistics for {label_name}...")
    
    # Sample 100 images
    sample_paths = paths[:100]
    if not sample_paths:
        print("   No images found.")
        return
        
    sizes = []
    means = []
    
    for p in tqdm(sample_paths):
        img = cv2.imread(p)
        if img is not None:
            sizes.append(img.shape)
            means.append(np.mean(img))
            
    # Check sizes
    unique_sizes = set(sizes)
    print(f"   Unique sizes: {unique_sizes}")
    if len(unique_sizes) > 1:
        print("   ⚠️  Images have different sizes.")
        
    # Check means
    avg_mean = np.mean(means)
    print(f"   Average Pixel Intensity: {avg_mean:.2f}")
    return avg_mean

def main():
    if not os.path.exists(CONFIG['dataset_root']):
        print(f"❌ Dataset root not found: {CONFIG['dataset_root']}")
        # Try to find it
        print("   Searching for 'Dataset' directory...")
        for root, dirs, files in os.walk("/Users/sameerkashyap/code"):
            if "Dataset" in dirs:
                print(f"   Found potential dataset at: {os.path.join(root, 'Dataset')}")
        return

    print(f"Loading dataset from {CONFIG['dataset_root']}...")
    try:
        datasets = load_dataset_paths(CONFIG['dataset_root'])
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    train_paths, train_labels = datasets['train']
    val_paths, val_labels = datasets['val']
    test_paths, test_labels = datasets['test']
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    # Check Leakage
    check_leakage(train_paths, val_paths)
    check_leakage(train_paths, test_paths)
    
    # Check Artifacts (Real vs Fake stats)
    # Split Train into Real/Fake
    train_real = [p for p, l in zip(train_paths, train_labels) if l == 0]
    train_fake = [p for p, l in zip(train_paths, train_labels) if l == 1]
    
    print(f"\nReal Images: {len(train_real)}, Fake Images: {len(train_fake)}")
    
    real_mean = check_image_stats(train_real, "Real")
    fake_mean = check_image_stats(train_fake, "Fake")
    
    diff = abs(real_mean - fake_mean)
    print(f"\nDifference in Mean Intensity: {diff:.2f}")
    if diff > 20:
        print("⚠️  LARGE difference in brightness. Model might be learning brightness!")

if __name__ == "__main__":
    main()
