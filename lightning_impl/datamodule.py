"""
PyTorch Lightning DataModule for CTD-FusionNet.
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import DeepfakeDataset, load_dataset_paths
from transforms import get_transforms

class DeepfakeDataModule(pl.LightningDataModule):
    def __init__(self, 
                 dataset_root, 
                 img_size=224, 
                 batch_size=32, 
                 num_workers=2, 
                 augmentation_config=None,
                 debug=False):
        super().__init__()
        self.dataset_root = dataset_root
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation_config = augmentation_config
        self.debug = debug
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        # Load dataset paths
        datasets = load_dataset_paths(self.dataset_root)
        
        X_train, y_train = datasets['train']
        X_val, y_val = datasets['val']
        X_test, y_test = datasets['test']
        
        # Log dataset statistics
        print("-" * 40)
        print(f"📊 Dataset Statistics:")
        print(f"   Train: {len(X_train)} samples ({sum(y_train)} Fake, {len(y_train)-sum(y_train)} Real)")
        print(f"   Val:   {len(X_val)} samples ({sum(y_val)} Fake, {len(y_val)-sum(y_val)} Real)")
        print(f"   Test:  {len(X_test)} samples ({sum(y_test)} Fake, {len(y_test)-sum(y_test)} Real)")
        print("-" * 40)

        if self.debug:
            print(f"[DEBUG] Subsampling for debugging...")
            
            def get_balanced_subset(X, y, n_per_class):
                indices_0 = [i for i, label in enumerate(y) if label == 0]
                indices_1 = [i for i, label in enumerate(y) if label == 1]
                
                selected_indices = indices_0[:n_per_class] + indices_1[:n_per_class]
                
                return [X[i] for i in selected_indices], [y[i] for i in selected_indices]

            # Subsample with balance
            X_train, y_train = get_balanced_subset(X_train, y_train, 25)
            X_val, y_val = get_balanced_subset(X_val, y_val, 15)
            X_test, y_test = get_balanced_subset(X_test, y_test, 15)
            
            print(f"[DEBUG] Subsampled sizes:")
            print(f"   Train: {len(X_train)} ({sum(y_train)} Fake)")
            print(f"   Val:   {len(X_val)} ({sum(y_val)} Fake)")
            print(f"   Test:  {len(X_test)} ({sum(y_test)} Fake)")

        # Create transforms
        train_transform = get_transforms(
            img_size=self.img_size,
            augmentation_config=self.augmentation_config,
            is_train=True
        )
        val_transform = get_transforms(
            img_size=self.img_size,
            augmentation_config=self.augmentation_config,
            is_train=False
        )
        
        # Create datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = DeepfakeDataset(X_train, y_train, train_transform)
            self.val_dataset = DeepfakeDataset(X_val, y_val, val_transform)
            
        if stage == 'test' or stage is None:
            self.test_dataset = DeepfakeDataset(X_test, y_test, val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False, # MPS doesn't support pin_memory well yet
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=True if self.num_workers > 0 else False
        )
