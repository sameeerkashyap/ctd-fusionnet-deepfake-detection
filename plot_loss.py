import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics(log_dir):
    metrics_path = os.path.join(log_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        print(f"No metrics found at {metrics_path}")
        return

    df = pd.read_csv(metrics_path)
    
    # Filter out steps where train_loss_epoch is NaN (it's logged at epoch end)
    # PyTorch Lightning logs step-wise and epoch-wise metrics in the same file
    
    # Plot Training and Validation Loss
    plt.figure(figsize=(10, 6))
    
    # Train Loss (Epoch)
    train_epoch_data = df[df['train_loss_epoch'].notna()]
    if not train_epoch_data.empty:
        plt.plot(train_epoch_data['epoch'], train_epoch_data['train_loss_epoch'], label='Train Loss (Epoch)', marker='o')
        
    # Val Loss
    val_epoch_data = df[df['val_loss'].notna()]
    if not val_epoch_data.empty:
        plt.plot(val_epoch_data['epoch'], val_epoch_data['val_loss'], label='Val Loss', marker='x')
        
    plt.title(f'Training and Validation Loss - {log_dir}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    output_file = "training_loss_plot.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    # Using version_2 as it is the most complete available log
    plot_metrics("lightning_logs/version_2")
