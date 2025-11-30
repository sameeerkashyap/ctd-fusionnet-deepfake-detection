#!/usr/bin/env python3
"""
Training script for Deepfake Classification (Binary)
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from pathlib import Path

from model import DeepfakeClassifier
from datamodule import DeepfakeDataModule


def main(args):
    print(f"\n{'#'*60}")
    print(f"STARTING DEEPFAKE TRAINING")
    print(f"{'#'*60}\n")
    
    # Seed
    pl.seed_everything(args.seed, workers=True)
    
    # Device
    print(f"{'='*60}")
    print(f"DEVICE INFORMATION")
    print(f"{'='*60}")
    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"MPS: Apple Silicon GPU")
    else:
        print(f"CPU: Using CPU")
    print(f"{'='*60}\n")
    
    # Output dirs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data
    data_module = DeepfakeDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size
    )
    
    # Model
    model = DeepfakeClassifier(
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename='best-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=args.early_stopping_patience,
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    progress_bar = RichProgressBar()
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir / "logs",
        name='deepfake_classification',
        version=args.experiment_name
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        devices='auto',
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor, progress_bar],
        log_every_n_steps=10,
        val_check_interval=1.0,  # Run validation every epoch
        check_val_every_n_epoch=1,
        precision=args.precision
    )
    
    # Train
    print(f"\n{'#'*60}")
    print(f"STARTING TRAINING LOOP")
    print(f"{'#'*60}\n")
    
    trainer.fit(model, data_module)
    
    # Test
    if args.run_test:
        print(f"\n{'#'*60}")
        print(f"STARTING TESTING")
        print(f"{'#'*60}\n")
        trainer.test(model, data_module, ckpt_path='best')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument("--data_dir", type=str, default="../ddata", 
                        help="Path to dataset directory (should contain Train/, Val/, Test/ with 0_real/ and 1_fake/ subdirectories)")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes (2 for binary)")
    parser.add_argument("--image_size", type=int, default=224)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--experiment_name", type=str, default="binary_diffusion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_test", action="store_true", default=True)
    
    args = parser.parse_args()
    main(args)
