
import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys

# Add parent directory to path to import dataset module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import load_dataset_paths

def get_debug_transforms(img_size=224, is_train=True):
    """
    Get simple transforms for debugging.
    """
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

class DebugDataset(Dataset):
    """
    Simplified dataset for debugging.
    Returns (image, label).
    """
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Load image
        img_bgr = cv2.imread(self.paths[idx])
        if img_bgr is None:
            raise ValueError(f"Failed to load image: {self.paths[idx]}")
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=img_rgb)
            img_tensor = transformed['image']
        else:
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()

        return img_tensor, torch.tensor(self.labels[idx]).long()
