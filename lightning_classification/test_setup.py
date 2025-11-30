#!/usr/bin/env python3
"""
Quick test script for Deepfake Classification Setup
"""

import torch
from model import DeepfakeClassifier
from datamodule import DeepfakeDataModule
import os

def test_setup():
    print("\n" + "="*60)
    print("TESTING DEEPFAKE CLASSIFICATION SETUP")
    print("="*60)
    
    # Check Data Directory
    data_dir = "../Diffusion"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return
    print(f"✓ Data directory found: {data_dir}")
    
    # Initialize Model
    print("\nInitializing Model...")
    model = DeepfakeClassifier(num_classes=2)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"✓ Model forward pass successful. Output shape: {output.shape}")
    
    # Initialize DataModule
    print("\nInitializing DataModule...")
    dm = DeepfakeDataModule(data_dir=data_dir, batch_size=4, num_workers=0)
    try:
        dm.setup(stage='fit')
        print("✓ DataModule setup successful")
        
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        print(f"✓ Train loader works. Batch shape: {batch[0].shape}, Labels: {batch[1]}")
        
    except Exception as e:
        print(f"❌ DataModule failed: {str(e)}")
        return

    print("\n" + "="*60)
    print("✅ SETUP VERIFIED SUCCESSFULLY")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_setup()
