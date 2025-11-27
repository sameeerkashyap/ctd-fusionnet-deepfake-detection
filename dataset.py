"""
Deepfake Dataset Module
Contains the DeepfakeDataset class and helper functions for loading and processing images.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
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
    Find Train, Val, and Test directories in the dataset root.
    
    Args:
        base_dir: Root directory of the dataset
    
    Returns:
        Tuple of (train_dir, val_dir, test_dir) paths
    """
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
    
    Args:
        dataset_root: Root directory containing Train/Val/Test folders
    
    Returns:
        Dictionary with keys: 'train', 'val', 'test'
        Each value is a tuple of (paths, labels) lists
    """
    train_dir, val_dir, test_dir = find_dataset_dirs(dataset_root)
    
    # Load Train set
    train_real = list_images(os.path.join(train_dir, 'Real'))
    train_fake = list_images(os.path.join(train_dir, 'Fake'))
    X_train = train_real + train_fake
    y_train = [0] * len(train_real) + [1] * len(train_fake)
    
    # Load Val set
    val_real = list_images(os.path.join(val_dir, 'Real'))
    val_fake = list_images(os.path.join(val_dir, 'Fake'))
    X_val = val_real + val_fake
    y_val = [0] * len(val_real) + [1] * len(val_fake)
    
    # Load Test set
    test_real = list_images(os.path.join(test_dir, 'Real'))
    test_fake = list_images(os.path.join(test_dir, 'Fake'))
    X_test = test_real + test_fake
    y_test = [0] * len(test_real) + [1] * len(test_fake)
    
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

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Compute CTD residual noise
        noise = fast_ctd_residual(img_rgb)
        
        # Apply transforms (both image and noise)
        transformed = self.transform(image=img_rgb, noise=noise)
        
        return transformed['image'], transformed['noise'], torch.tensor(self.labels[idx]).long()


