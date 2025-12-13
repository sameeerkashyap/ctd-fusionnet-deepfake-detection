"""
Training script for deepfake detection using CTD-FusionNet with PyTorch Lightning.
"""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning_module import FusionNetLightning
from datamodule import DeepfakeDataModule

# Configuration
CONFIG = {
    'dataset_root': "data",
    'img_size': 224,
    'batch_size': 32,
    'num_workers': 2,
    'learning_rate': 1e-4,
    'num_epochs': 5,
    'weight_decay': 1e-3,
    'spsl_model_name': "resnet18",  # Lighter backbone
    'debug': False,  # Set to True for debugging (subsampling, prints)
    'augmentation': {
        'crop_scale': (0.75, 1.0),
        'horizontal_flip_p': 0.5,
        'gauss_noise_p': 0.15,
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
       'mean': (0.519982, 0.425405, 0.379991),
       'std': (0.279140, 0.254101, 0.255005)
    }
    }
}

def main():
    # Set seed for reproducibility
    pl.seed_everything(42)

    # Check for MPS
    if torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
        print("🚀 Using Mac MPS (Metal Performance Shaders) acceleration")
    elif torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
        print("🚀 Using CUDA acceleration")
    else:
        accelerator = "cpu"
        devices = "auto"
        print("⚠️ Using CPU (Slow)")

    # Initialize DataModule
    dm = DeepfakeDataModule(
        dataset_root=CONFIG['dataset_root'],
        img_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        augmentation_config=CONFIG['augmentation'],
        debug=CONFIG['debug']
    )

    # Initialize Model
    model = FusionNetLightning(
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        num_epochs=CONFIG['num_epochs'],
        spsl_model_name=CONFIG['spsl_model_name'],
        debug=CONFIG['debug']
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='fusionnet-{epoch:02d}-{val_auc:.4f}',
        monitor='val_auc',
        mode='max',
        save_top_k=1,
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_auc',
        patience=3,
        mode='max'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=CONFIG['num_epochs'],
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        log_every_n_steps=10,
        fast_dev_run=False  # Set to True for a quick 1-epoch dry run
    )

    # Train
    print("\n🚀 Starting Training...")
    trainer.fit(model, dm)

    # Test
    print("\n🧪 Starting Testing...")
    trainer.test(model, dm, ckpt_path='best')

if __name__ == "__main__":
    main()
