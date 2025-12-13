"""
Deepfake Dataset Module
Contains the DeepfakeDataset class and helper functions for loading and processing images.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
try:
    from .transforms import get_transforms
except ImportError:
    from transforms import get_transforms


def fast_ctd_residual(img_rgb_uint8):
    """
    Compute CTD (Copy-Move Tampering Detection) residual noise from RGB image.
    
    Args:
        img_rgb_uint8: RGB image as uint8 numpy array (H, W, 3)
    
    Returns:
        noise: Residual noise image as uint8 numpy array (H, W, 3)
    """
    den = cv2.GaussianBlur(img_rgb_uint8, (5, 5), 0)
    noise = np.clip(img_rgb_uint8 - den + 128, 0, 255).astype(np.uint8)
    return noise


def list_images(folder):
    """
    List all image files in a folder.
    
    Args:
        folder: Path to folder containing images
    
    Returns:
        List of full paths to image files
    """
    if not os.path.exists(folder):
        return []
    return sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    ])


def find_dataset_dirs(base_dir):
    """
    Find Train, Val, and Test directories in the dataset root (case-insensitive).
    
    Args:
        base_dir: Root directory of the dataset
    
    Returns:
        Tuple of (train_dir, val_dir, test_dir) paths
    """
    train_dir = None
    val_dir = None
    test_dir = None
    
    # Check for both capitalized and lowercase versions
    for root, dirs, _ in os.walk(base_dir):
        # Look for direct children first to avoid deep recursion issues if structure is flat
        lower_dirs = {d.lower(): d for d in dirs}
        
        if 'train' in lower_dirs:
            train_dir = os.path.join(root, lower_dirs['train'])
        if 'val' in lower_dirs:
            val_dir = os.path.join(root, lower_dirs['val'])
        if 'test' in lower_dirs:
            test_dir = os.path.join(root, lower_dirs['test'])
            
        # If we found all three in the top level, break. 
        # Otherwise os.walk will continue, which might be what we want if they are nested.
        # But usually they are at the top level of base_dir.
        if train_dir and val_dir and test_dir:
            break
            
    # Fallback: if os.walk didn't find them (e.g. if base_dir IS the parent), check directly
    if not train_dir:
        if os.path.exists(os.path.join(base_dir, 'Train')): train_dir = os.path.join(base_dir, 'Train')
        elif os.path.exists(os.path.join(base_dir, 'train')): train_dir = os.path.join(base_dir, 'train')
        
    if not val_dir:
        if os.path.exists(os.path.join(base_dir, 'Val')): val_dir = os.path.join(base_dir, 'Val')
        elif os.path.exists(os.path.join(base_dir, 'val')): val_dir = os.path.join(base_dir, 'val')
        
    if not test_dir:
        if os.path.exists(os.path.join(base_dir, 'Test')): test_dir = os.path.join(base_dir, 'Test')
        elif os.path.exists(os.path.join(base_dir, 'test')): test_dir = os.path.join(base_dir, 'test')
    
    if train_dir is None:
        raise FileNotFoundError(f"Train directory not found in {base_dir}")
    if val_dir is None:
        raise FileNotFoundError(f"Val directory not found in {base_dir}")
    if test_dir is None:
        raise FileNotFoundError(f"Test directory not found in {base_dir}")
    
    return train_dir, val_dir, test_dir


def load_dataset_paths(dataset_root):
    """
    Load image paths and labels from Train, Val, and Test directories.
    Supports arbitrary classes (inferred from subdirectories).
    
    Args:
        dataset_root: Root directory containing Train/Val/Test folders
    
    Returns:
        Dictionary with keys: 'train', 'val', 'test'
        Each value is a tuple of (paths, labels) lists
    """
    train_dir, val_dir, test_dir = find_dataset_dirs(dataset_root)
    
    def load_from_dir(directory, class_to_idx=None):
        classes = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
        
        # If class_to_idx is provided, verify classes match. If not, create it.
        if class_to_idx is None:
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        paths = []
        labels = []
        
        for cls_name in classes:
            if cls_name not in class_to_idx:
                continue # Skip unknown classes if we are enforcing a schema, or maybe error out?
                
            cls_dir = os.path.join(directory, cls_name)
            img_paths = list_images(cls_dir)
            paths.extend(img_paths)
            labels.extend([class_to_idx[cls_name]] * len(img_paths))
            
        return paths, labels, class_to_idx

    # Load Train set and determine classes
    X_train, y_train, class_to_idx = load_from_dir(train_dir)
    print(f"Detected classes: {class_to_idx}")
    
    # Load Val set using same classes
    X_val, y_val, _ = load_from_dir(val_dir, class_to_idx)
    
    # Load Test set using same classes
    X_test, y_test, _ = load_from_dir(test_dir, class_to_idx)
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }


class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for deepfake detection.
    
    Returns tuples of (image, noise, label) where:
    - image: RGB image tensor after augmentation
    - noise: CTD residual noise tensor after augmentation
    - label: 0 for Real, 1 for Fake
    """
    
    def __init__(self, paths, labels, transform):
        """
        Initialize the dataset.
        
        Args:
            paths: List of image file paths
            labels: List of labels (0=Real, 1=Fake)
            transform: Albumentations transform pipeline
        """
        self.paths = paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Tuple of (image_tensor, noise_tensor, label_tensor)
        """
        # Load image
        img_bgr = cv2.imread(self.paths[idx])
        if img_bgr is None:
             raise ValueError(f"Failed to load image at {self.paths[idx]}")

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Compute CTD residual noise
        noise = fast_ctd_residual(img_rgb)
        
        # Apply transforms (both image and noise)
        transformed = self.transform(image=img_rgb, noise=noise)
        
        return transformed['image'], transformed['noise'], torch.tensor(self.labels[idx]).long()
