import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm

from lightning_impl.lightning_module import FusionNetLightning
from lightning_impl.dataset import load_dataset_paths, DeepfakeDataset
from lightning_impl.transforms import get_transforms

def generate_confusion_matrix(checkpoint_path, data_root="data", batch_size=32, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # 1. Load Model
    print(f"Loading model from {checkpoint_path}...")
    try:
        model = FusionNetLightning.load_from_checkpoint(checkpoint_path, map_location=device)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 2. Load Data
    print(f"Loading validation data from {data_root}...")
    try:
        data_splits = load_dataset_paths(data_root)
        X_val, y_val = data_splits['val']
        
        # Check if we have data
        if not X_val:
            print("No validation data found.")
            return
            
        print(f"Found {len(X_val)} validation samples.")
        
        # Create Dataset and DataLoader
        val_transforms = get_transforms(is_train=False)
        val_dataset = DeepfakeDataset(X_val, y_val, val_transforms)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # 3. Inference
    all_preds = []
    all_labels = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            imgs, noises, labels = batch
            imgs = imgs.to(device)
            noises = noises.to(device)
            
            logits = model(imgs, noises)
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(probs, 1)

            # Invert predictions because model classes are inverted relative to dataset
            # Model: 0=Real, 1=Fake
            # Dataset: 0=Fake, 1=Real
            # So if model predicts 0 (Real), we want it to map to 1 (Real)
            # If model predicts 1 (Fake), we want it to map to 0 (Fake)
            # Actually:
            # Model 0 -> Real (Dataset 1)
            # Model 1 -> Fake (Dataset 0)
            # So Pred 0 -> 1, Pred 1 -> 0
            # This is equivalent to 1 - pred
            preds = 1 - preds
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # 4. Compute Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    
    # Classes: 0=Fake, 1=Real (based on alphabetical order of 'fake', 'real')
    class_names = ['Fake', 'Real']
    
    # 5. Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix\n{os.path.basename(checkpoint_path)}')
    
    output_file = "confusion_matrix.png"
    plt.savefig(output_file)
    print(f"Confusion matrix saved to {output_file}")
    
    # Print metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\nMetrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    ckpt_path = "checkpoints-inverted/fusionnet-epoch=08-val_auc=1.0000.ckpt"
    generate_confusion_matrix(ckpt_path)
