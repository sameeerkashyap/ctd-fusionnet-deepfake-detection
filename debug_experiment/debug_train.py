
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from collections import Counter
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_experiment.debug_dataset import DebugDataset, get_debug_transforms
from debug_experiment.debug_model import SimpleClassifier
from dataset import load_dataset_paths

# Configuration
CONFIG = {
    'dataset_root': "/Users/sameerkashyap/code/ctd-fusionnet-deepfake-detection/CatandDog/split_dataset",
    'img_size': 224,
    'batch_size': 32,
    'num_workers': 2,
    'learning_rate': 1e-4,
    'num_epochs': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
    'save_dir': 'debug_experiment/results'
}

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

def plot_metrics(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()
    
    # Accuracy
    plt.figure()
    plt.plot(epochs, history['val_acc'], label='Val Accuracy')
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'))
    plt.close()

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        
        # Ensure outputs are the right shape for BCE loss: [batch_size, 1] or [batch_size]
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(1)
        
        # Convert labels to float for BCE loss (0.0 or 1.0)
        labels_float = labels.float().unsqueeze(1)
        loss = criterion(outputs, labels_float)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device, desc="Validating"):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    
    pbar = tqdm(loader, desc=desc, leave=False)
    with torch.no_grad():
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            outputs = model(imgs)
            
            # Ensure outputs are the right shape for BCE loss: [batch_size, 1] or [batch_size]
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            
            # Convert labels to float for BCE loss (0.0 or 1.0)
            labels_float = labels.float().unsqueeze(1)
            loss = criterion(outputs, labels_float)
            total_loss += loss.item()
            
            # For binary classification: apply sigmoid to get probability of class 1
            # outputs shape: [batch_size, 1], squeeze to [batch_size]
            probs = torch.sigmoid(outputs.squeeze(-1))  # Only squeeze last dimension
            
            # Handle single item batches
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
            
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            
    avg_loss = total_loss / len(loader)
    
    # Metrics
    preds = [1 if p > 0.5 else 0 for p in all_probs]
    
    # Debug: Check for suspicious patterns
    if len(all_probs) > 0:
        min_prob = min(all_probs)
        max_prob = max(all_probs)
        mean_prob = np.mean(all_probs)
        prob_std = np.std(all_probs)
        
        # Check if all predictions are the same
        unique_preds = set(preds)
        unique_labels = set(all_labels)
        
        # Print debug info if accuracy is 100%
        accuracy = accuracy_score(all_labels, preds)
        if accuracy >= 0.99:  # 99% or higher
            print(f"\n⚠️  DEBUG: Suspiciously high accuracy ({accuracy:.4f})")
            print(f"   Probabilities: min={min_prob:.4f}, max={max_prob:.4f}, mean={mean_prob:.4f}, std={prob_std:.4f}")
            print(f"   Unique predictions: {unique_preds}")
            print(f"   Unique labels: {unique_labels}")
            print(f"   Label distribution: {Counter(all_labels)}")
            print(f"   Prediction distribution: {Counter(preds)}")
            
            # Check if model is always predicting one class
            if len(unique_preds) == 1:
                print(f"   ❌ MODEL IS ALWAYS PREDICTING CLASS {list(unique_preds)[0]}!")
                print(f"      This suggests the model hasn't learned properly.")
            
            # Sample some predictions vs labels
            print(f"   Sample predictions (first 10):")
            for i in range(min(10, len(preds))):
                print(f"      Label: {all_labels[i]}, Pred: {preds[i]}, Prob: {all_probs[i]:.4f}")
    
    metrics = {
        'loss': avg_loss,
        'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5,
        'accuracy': accuracy_score(all_labels, preds),
        'precision': precision_score(all_labels, preds, zero_division=0),
        'recall': recall_score(all_labels, preds, zero_division=0),
        'f1': f1_score(all_labels, preds, zero_division=0),
        'labels': all_labels,
        'preds': preds
    }
    
    return metrics

def main():
    print(f"Using device: {CONFIG['device']}")
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    # Load Data
    print("Loading dataset paths...")
    datasets = load_dataset_paths(CONFIG['dataset_root'])
    
    train_paths, train_labels = datasets['train']
    val_paths, val_labels = datasets['val']
    test_paths, test_labels = datasets['test']
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    # Create Datasets
    train_ds = DebugDataset(train_paths, train_labels, get_debug_transforms(CONFIG['img_size'], is_train=True))
    val_ds = DebugDataset(val_paths, val_labels, get_debug_transforms(CONFIG['img_size'], is_train=False))
    test_ds = DebugDataset(test_paths, test_labels, get_debug_transforms(CONFIG['img_size'], is_train=False))
    
    # Create Loaders
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    
    # Model
    print("Initializing ResNet18...")
    model = SimpleClassifier().to(CONFIG['device'])
    # For binary classification with single output, use BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Training Loop
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    print("Starting training...")
    for epoch in range(CONFIG['num_epochs']):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
        
        # Evaluate both train and validation metrics
        train_metrics = evaluate(model, train_loader, criterion, CONFIG['device'], desc="Evaluating Train")
        val_metrics = evaluate(model, val_loader, criterion, CONFIG['device'])
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        print(f"  Train - Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f} | "
              f"Precision: {train_metrics['precision']:.4f} | Recall: {train_metrics['recall']:.4f} | "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
              f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")
              
        if val_metrics['accuracy'] == 1.0:
            print("  ⚠️  WARNING: Validation Accuracy hit 100%. This is suspicious.")
            
    # Plot training history
    plot_metrics(history, CONFIG['save_dir'])
    
    # Final Test
    print("\nRunning Final Test...")
    test_metrics = evaluate(model, test_loader, criterion, CONFIG['device'], desc="Testing")
    
    print("="*40)
    print("FINAL TEST RESULTS")
    print("="*40)
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"AUC:       {test_metrics['auc']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1 Score:  {test_metrics['f1']:.4f}")
    print("="*40)
    
    # Confusion Matrix
    plot_confusion_matrix(test_metrics['labels'], test_metrics['preds'], 
                         os.path.join(CONFIG['save_dir'], 'confusion_matrix.png'))
    print(f"Results and plots saved to {CONFIG['save_dir']}")

if __name__ == "__main__":
    main()
