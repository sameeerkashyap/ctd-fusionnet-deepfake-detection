
import os
import shutil
import random
from collections import defaultdict

def resplit_dataset_by_video(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Re-split dataset ensuring no video appears in multiple splits.
    Assumes filenames like: videoID_frameXXX.jpg
    """
    print("🔄 Re-splitting dataset by video ID...")
    
    # Group files by video ID
    video_groups = defaultdict(list)
    
    for split in ['Train', 'Val', 'Test']:
        for class_name in ['Real', 'Fake']:
            folder = os.path.join(source_dir, split, class_name)
            if not os.path.exists(folder):
                continue
                
            for fname in os.listdir(folder):
                if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue
                    
                # Extract video ID (customize this based on your naming)
                video_id = "_".join(fname.split('_')[:2])  # e.g., "video01_frame10.jpg" -> "video01"
                
                full_path = os.path.join(folder, fname)
                video_groups[(class_name, video_id)].append(full_path)
    
    print(f"Found {len(video_groups)} unique video groups")
    
    # Split video IDs (not frames)
    all_video_ids = list(video_groups.keys())
    random.shuffle(all_video_ids)
    
    n_train = int(len(all_video_ids) * train_ratio)
    n_val = int(len(all_video_ids) * val_ratio)
    
    train_ids = all_video_ids[:n_train]
    val_ids = all_video_ids[n_train:n_train+n_val]
    test_ids = all_video_ids[n_train+n_val:]
    
    # Copy files to new structure
    for split_name, video_ids in [('Train', train_ids), ('Val', val_ids), ('Test', test_ids)]:
        for class_name, video_id in video_ids:
            files = video_groups[(class_name, video_id)]
            
            dest_folder = os.path.join(output_dir, split_name, class_name)
            os.makedirs(dest_folder, exist_ok=True)
            
            for src_file in files:
                dest_file = os.path.join(dest_folder, os.path.basename(src_file))
                shutil.copy2(src_file, dest_file)
    
    print(f"✅ Dataset re-split complete!")
    print(f"   Train: {len(train_ids)} videos")
    print(f"   Val: {len(val_ids)} videos")
    print(f"   Test: {len(test_ids)} videos")

if __name__ == "__main__":
    # CONFIGURE THESE PATHS
    SOURCE_DIR = "/path/to/original/Dataset"
    OUTPUT_DIR = "/path/to/new/Dataset_NoLeakage"
    
    resplit_dataset_by_video(SOURCE_DIR, OUTPUT_DIR)
