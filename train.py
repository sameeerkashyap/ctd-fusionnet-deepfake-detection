"""
Training script for deepfake detection using CTD-FusionNet.
All configuration parameters are centralized in the CONFIG dict.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from dataset import DeepfakeDataset, load_dataset_paths
from transforms import get_transforms
from model import FusionNetCTD

# Configuration dictionary - controls all training parameters
CONFIG = {
    # Dataset parameters
    'dataset_root': "../Dataset",
    'img_size': 224,

    # DataLoader parameters
    'batch_size': 32,
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

    # Training parameters
    'learning_rate': 1e-4,
    'num_epochs': 10,
    'weight_decay': 1e-4,

    # Device
    'device': 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'),

    # Model saving
    'save_dir': 'checkpoints',
    'save_best_only': True,

    # Debug mode (subsamples dataset for quick testing)
    'debug': False
}




def create_dataloaders(config):
    """
    Create train, validation, and test dataloaders.

    Args:
        config: Configuration dictionary

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load dataset paths
    datasets = load_dataset_paths(config['dataset_root'])

    X_train, y_train = datasets['train']
    X_val, y_val = datasets['val']
    X_test, y_test = datasets['test']

    # Debug mode: subsample datasets for quick testing
    if config['debug']:
        # Take very small samples for ultra-fast debugging
        train_sample = min(50, len(X_train))  # Max 50 samples total for training
        val_sample = min(25, len(X_val))      # Max 25 samples for validation
        test_sample = min(25, len(X_test))    # Max 25 samples for testing

        # Sample with stratification (maintain class balance)
        train_indices = []
        for label in [0, 1]:
            label_indices = [i for i, y in enumerate(y_train) if y == label]
            sample_size = min(train_sample // 2, len(label_indices))
            train_indices.extend(label_indices[:sample_size])

        val_indices = []
        for label in [0, 1]:
            label_indices = [i for i, y in enumerate(y_val) if y == label]
            sample_size = min(val_sample // 2, len(label_indices))
            val_indices.extend(label_indices[:sample_size])

        test_indices = []
        for label in [0, 1]:
            label_indices = [i for i, y in enumerate(y_test) if y == label]
            sample_size = min(test_sample // 2, len(label_indices))
            test_indices.extend(label_indices[:sample_size])

        X_train = [X_train[i] for i in train_indices]
        y_train = [y_train[i] for i in train_indices]
        X_val = [X_val[i] for i in val_indices]
        y_val = [y_val[i] for i in val_indices]
        X_test = [X_test[i] for i in test_indices]
        y_test = [y_test[i] for i in test_indices]

        print(f"🐛 Debug mode: Using {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples")

    # Create transforms
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
    train_dataset = DeepfakeDataset(X_train, y_train, train_transform)
    val_dataset = DeepfakeDataset(X_val, y_val, val_transform)
    test_dataset = DeepfakeDataset(X_test, y_test, val_transform)

    # Create data loaders
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

    return train_loader, val_loader, test_loader


def create_model_and_optimizer(config):
    """
    Create model, loss function, and optimizer.

    Args:
        config: Configuration dictionary

    Returns:
        tuple: (model, criterion, optimizer, scheduler)
    """
    # Create model
    model = FusionNetCTD()
    model = model.to(config['device'])

    # Create loss function (CrossEntropyLoss)
    criterion = nn.CrossEntropyLoss()

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Create scheduler (cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs']
    )

    return model, criterion, optimizer, scheduler


def train_epoch(model, train_loader, criterion, optimizer, config):
    """
    Train for one epoch.

    Args:
        model: The model to train
        train_loader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        config: Configuration dictionary

    Returns:
        float: Average training loss
    """
    model.train()
    total_loss = 0.0

    # Create progress bar for training batches
    progress_bar = tqdm(train_loader, desc="Training", unit="batch")

    for batch_idx, (img, noise, labels) in enumerate(progress_bar):
        img = img.to(config['device'])
        noise = noise.to(config['device'])
        labels = labels.to(config['device'])

        optimizer.zero_grad()
        outputs = model(img, noise)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

        # Update progress bar with current loss
        current_loss = loss.item()
        progress_bar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
        })

    progress_bar.close()
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, config):
    """
    Validate the model.

    Args:
        model: The model to validate
        val_loader: Validation dataloader
        criterion: Loss function
        config: Configuration dictionary

    Returns:
        tuple: (val_loss, auc, accuracy, precision, recall, f1)
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    # Create progress bar for validation batches
    progress_bar = tqdm(val_loader, desc="Validating", unit="batch")

    with torch.no_grad():
        for img, noise, labels in progress_bar:
            img = img.to(config['device'])
            noise = noise.to(config['device'])
            labels = labels.to(config['device'])

            outputs = model(img, noise)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    progress_bar.close()

    val_loss = total_loss / len(val_loader)
    auc = roc_auc_score(all_labels, all_probs)
    accuracy = accuracy_score(all_labels, [p > 0.5 for p in all_probs])
    precision = precision_score(all_labels, [p > 0.5 for p in all_probs])
    recall = recall_score(all_labels, [p > 0.5 for p in all_probs])
    f1 = f1_score(all_labels, [p > 0.5 for p in all_probs])

    return val_loss, auc, accuracy, precision, recall, f1


def test_model(model, test_loader, config):
    """
    Test the model on test set.

    Args:
        model: The trained model
        test_loader: Test dataloader
        config: Configuration dictionary

    Returns:
        dict: Test metrics
    """
    model.eval()
    all_labels = []
    all_probs = []

    # Create progress bar for test batches
    progress_bar = tqdm(test_loader, desc="Testing", unit="batch")

    with torch.no_grad():
        for img, noise, labels in progress_bar:
            img = img.to(config['device'])
            noise = noise.to(config['device'])
            labels = labels.to(config['device'])

            outputs = model(img, noise)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    progress_bar.close()

    # Calculate metrics
    auc = roc_auc_score(all_labels, all_probs)
    accuracy = accuracy_score(all_labels, [p > 0.5 for p in all_probs])
    f1 = f1_score(all_labels, [p > 0.5 for p in all_probs])
    precision = precision_score(all_labels, [p > 0.5 for p in all_probs])
    recall = recall_score(all_labels, [p > 0.5 for p in all_probs])

    return {
        'auc': auc,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def test_checkpoint(checkpoint_path, config=CONFIG):
    """
    Load and test a saved model checkpoint.

    Args:
        checkpoint_path: Path to the saved checkpoint
        config: Configuration dictionary

    Returns:
        dict: Test metrics
    """
    print("=" * 80)
    print("Testing Model Checkpoint")
    print("=" * 80)
    print(f"📁 Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    saved_config = checkpoint.get('config', config)

    # Create test dataloader
    print("\n🔄 Creating test dataloader...")
    _, _, test_loader = create_dataloaders(saved_config)
    print(f"  ✅ Test loader: {len(test_loader)} batches")

    # Create model and load state
    print("\n🏗️  Loading model from checkpoint...")
    model = FusionNetCTD()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(saved_config['device'])

    # Show checkpoint info
    print(f"  ✅ Model loaded from epoch {checkpoint['epoch']}")
    print(f"  📊 Validation AUC at save: {checkpoint.get('val_auc', 'N/A'):.4f}")

    # Test the model
    print("\n🧪 Testing model...")
    test_metrics = test_model(model, test_loader, saved_config)

    print("\n📊 Test Results:")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")

    print("\n" + "=" * 80)
    print("Testing completed successfully!")
    print("=" * 80)

    return test_metrics


def train_model(config=CONFIG):
    """
    Main training function.

    Args:
        config: Configuration dictionary
    """
    print("=" * 80)
    print("Training CTD-FusionNet")
    print("=" * 80)

    # Create dataloaders
    print("\n🔄 Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    print(f"  ✅ Train: {len(train_loader)} batches")
    print(f"  ✅ Val: {len(val_loader)} batches")
    print(f"  ✅ Test: {len(test_loader)} batches")

    # Create model, loss, optimizer
    print("\n🏗️  Creating model and optimizer...")
    model, criterion, optimizer, scheduler = create_model_and_optimizer(config)
    print(f"  ✅ Model: CrossEntropyLoss")
    print(f"  ✅ Device: {config['device']}")

    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)

    # Training loop
    best_auc = 0.0
    print("\n🚀 Starting training...")
    print(f"  📊 Epochs: {config['num_epochs']}")
    print(f"  📊 Batch size: {config['batch_size']}")
    print(f"  📊 Learning rate: {config['learning_rate']}")

    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config)

        # Validate
        val_loss, val_auc, val_acc, val_precision, val_recall, val_f1 = validate(model, val_loader, criterion, config)

        # Update scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start_time

        print(f"Epoch {epoch+1:2d}/{config['num_epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val F1: {val_f1:.4f} | "
              f"Time: {epoch_time:.1f}s | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            checkpoint_path = os.path.join(config['save_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1,
                'config': config
            }, checkpoint_path)
            print(f"💾 Saved best model (AUC: {val_auc:.4f})")

    print("\n🏁 Training completed!")
    print(f"📊 Best validation AUC: {best_auc:.4f}")
    print(f"💾 Best model saved at: {checkpoint_path}")

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)

    return checkpoint_path


if __name__ == "__main__":
    # Step 1: Train the model
    print("🚀 Starting CTD-FusionNet Training Pipeline")
    print("=" * 80)

    checkpoint_path = train_model(config=CONFIG)

    # Step 2: Test the trained model
    print("\n" + "=" * 80)
    print("📋 Moving to Testing Phase")
    print("=" * 80)

    test_metrics = test_checkpoint(checkpoint_path, config=CONFIG)

    print("\n🎉 Pipeline completed successfully!")
    print(f"📊 Final Test AUC: {test_metrics['auc']:.4f}")
    print(f"📊 Final Test F1: {test_metrics['f1']:.4f}")

