"""
Training script for deepfake detection using CTD-FusionNet.
All configuration parameters are centralized in the CONFIG dict.
"""

import os
import torch
from torch.utils.data import DataLoader
from dataset import DeepfakeDataset, load_dataset_paths
from transforms import get_transforms

# Configuration dictionary - controls all training parameters
CONFIG = {
    # Dataset parameters
    'dataset_root': "../Dataset",
    'img_size': 224,

    # DataLoader parameters
    'batch_size': 4,
    'num_workers': 2,

    # Transform parameters
    'augmentation': {
        'crop_scale': (0.75, 1.0),  # RandomResizedCrop scale range
        'horizontal_flip_p': 0.5,    # Horizontal flip probability
        'gauss_noise_p': 0.15,       # Gaussian noise probability
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1,
            'p': 0.4
        },
        'compression': {
            'quality_range': (70, 100),
            'p': 0.2
        },
        'normalize': {
        'mean': (0.433168, 0.375803, 0.337967),
        'std': (0.270012, 0.247510, 0.239540)
   }
    },

    # Training parameters (for future use)
    'learning_rate': 1e-4,
    'num_epochs': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


def test_dataset(config=CONFIG):
    """
    Test the DeepfakeDataset by loading data and iterating through batches.

    Args:
        config: Configuration dictionary containing all parameters
    """
    print("=" * 80)
    print("Testing DeepfakeDataset")
    print("=" * 80)
    
    # Load dataset paths
    print(f"\n📁 Loading dataset from: {config['dataset_root']}")
    try:
        datasets = load_dataset_paths(config['dataset_root'])
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"\nPlease ensure the dataset directory structure is:")
        print(f"  {config['dataset_root']}/")
        print(f"    ├── Train/")
        print(f"    │   ├── Real/")
        print(f"    │   └── Fake/")
        print(f"    ├── Val/")
        print(f"    │   ├── Real/")
        print(f"    │   └── Fake/")
        print(f"    └── Test/")
        print(f"        ├── Real/")
        print(f"        └── Fake/")
        return
    
    X_train, y_train = datasets['train']
    X_val, y_val = datasets['val']
    X_test, y_test = datasets['test']
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Train: {len(X_train)} images ({y_train.count(0)} Real, {y_train.count(1)} Fake)")
    print(f"  Val:   {len(X_val)} images ({y_val.count(0)} Real, {y_val.count(1)} Fake)")
    print(f"  Test:  {len(X_test)} images ({y_test.count(0)} Real, {y_test.count(1)} Fake)")
    
    # Create transforms
    print(f"\n🔄 Creating transforms...")
    train_transform = get_transforms(
        img_size=config['img_size'],
        augmentation_config=config['augmentation'],
        is_train=True
    )
    val_transform = get_transforms(
        img_size=config['img_size'],
        augmentation_config=config['augmentation'],
        is_train=False
    )
    
    # Create datasets
    print(f"\n📦 Creating datasets...")
    train_dataset = DeepfakeDataset(X_train, y_train, train_transform)
    val_dataset = DeepfakeDataset(X_val, y_val, val_transform)
    test_dataset = DeepfakeDataset(X_test, y_test, val_transform)
    
    print(f"  ✅ Train dataset: {len(train_dataset)} samples")
    print(f"  ✅ Val dataset: {len(val_dataset)} samples")
    print(f"  ✅ Test dataset: {len(test_dataset)} samples")
    
    # Create data loaders
    print(f"\n🔄 Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['num_workers'] > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['num_workers'] > 0 else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['num_workers'] > 0 else False
    )
    
    print(f"  ✅ Train loader: {len(train_loader)} batches")
    print(f"  ✅ Val loader: {len(val_loader)} batches")
    print(f"  ✅ Test loader: {len(test_loader)} batches")
    
    # Test loading a batch
    print(f"\n🧪 Testing batch loading...")
    try:
        # Get a batch from train loader
        print(f"  Loading a batch from train loader...")
        img_batch, noise_batch, label_batch = next(iter(train_loader))
        
        print(f"  ✅ Successfully loaded batch!")
        print(f"    Image shape: {img_batch.shape}")
        print(f"    Noise shape: {noise_batch.shape}")
        print(f"    Label shape: {label_batch.shape}")
        print(f"    Label values: {label_batch.tolist()}")
        print(f"    Image dtype: {img_batch.dtype}")
        print(f"    Noise dtype: {noise_batch.dtype}")
        print(f"    Image range: [{img_batch.min():.3f}, {img_batch.max():.3f}]")
        print(f"    Noise range: [{noise_batch.min():.3f}, {noise_batch.max():.3f}]")
        
        # Test validation loader
        print(f"\n  Loading a batch from val loader...")
        img_val, noise_val, label_val = next(iter(val_loader))
        print(f"  ✅ Validation batch loaded successfully!")
        print(f"    Image shape: {img_val.shape}")
        print(f"    Noise shape: {noise_val.shape}")
        print(f"    Label shape: {label_val.shape}")
        
        # Test a few more batches to ensure consistency
        print(f"\n  Testing multiple batches...")
        for i, (img, noise, label) in enumerate(train_loader):
            if i >= 2:  # Test first 3 batches
                break
            print(f"    Batch {i+1}: img={img.shape}, noise={noise.shape}, labels={label.tolist()}")
        
        print(f"\n✅ All tests passed! Dataset is working correctly.")
        
    except Exception as e:
        print(f"❌ Error loading batch: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("Dataset test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    # Test the dataset using configuration
    test_dataset(config=CONFIG)

