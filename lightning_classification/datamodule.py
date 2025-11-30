import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Optional
import os
import torch

class DeepfakeDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Deepfake Classification (Binary: Real vs Fake)
    Expects data structure:
    data_dir/
      Train/
        0_real/
        1_fake/
      Val/
        ...
      Test/
        ...
    """
    
    def __init__(
        self,
        data_dir: str = "../Diffusion",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
        print(f"\n{'='*60}")
        print(f"INITIALIZING DEEPFAKE DATAMODULE")
        print(f"{'='*60}")
        print(f"Data directory: {data_dir}")
        print(f"Batch size: {batch_size}")
        print(f"Num workers: {num_workers}")
        print(f"Image size: {image_size}")
        print(f"{'='*60}\n")
        
        # Transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for each stage"""
        print(f"\n{'='*60}")
        print(f"SETUP STAGE: {stage}")
        print(f"{'='*60}")
        
        # Define paths
        train_dir = os.path.join(self.data_dir, 'Train')
        val_dir = os.path.join(self.data_dir, 'Val')
        test_dir = os.path.join(self.data_dir, 'Test')
        
        # Check if directories exist
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Train directory not found at {train_dir}")
        if not os.path.exists(val_dir):
            raise FileNotFoundError(f"Val directory not found at {val_dir}")
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test directory not found at {test_dir}")

        if stage == 'fit' or stage is None:
            print("Loading Training Set...")
            self.train_dataset = datasets.ImageFolder(
                root=train_dir,
                transform=self.train_transform
            )
            print(f"  ✓ Loaded {len(self.train_dataset)} training images")
            print(f"  ✓ Classes: {self.train_dataset.classes}")
            print(f"  ✓ Class to idx: {self.train_dataset.class_to_idx}")
            
            print("Loading Validation Set...")
            self.val_dataset = datasets.ImageFolder(
                root=val_dir,
                transform=self.val_test_transform
            )
            print(f"  ✓ Loaded {len(self.val_dataset)} validation images")
        
        if stage == 'test' or stage is None:
            print("Loading Test Set...")
            self.test_dataset = datasets.ImageFolder(
                root=test_dir,
                transform=self.val_test_transform
            )
            print(f"  ✓ Loaded {len(self.test_dataset)} test images")
        
        print(f"{'='*60}\n")
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
