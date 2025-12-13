import pandas as pd
import os
import glob

def inspect_logs():
    log_dir = "lightning_logs"
    versions = glob.glob(os.path.join(log_dir, "version_*"))
    
    print(f"Found {len(versions)} versions.")
    
    for version in sorted(versions):
        metrics_path = os.path.join(version, "metrics.csv")
        if not os.path.exists(metrics_path):
            print(f"{version}: No metrics.csv found")
            continue
            
        try:
            df = pd.read_csv(metrics_path)
            if 'epoch' not in df.columns:
                print(f"{version}: 'epoch' column not found")
                continue
                
            max_epoch = df['epoch'].max()
            
            val_auc = "N/A"
            if 'val_auc' in df.columns:
                val_auc = df['val_auc'].max()
                
            print(f"{version}: Max Epoch={max_epoch}, Max Val AUC={val_auc}")
            
        except Exception as e:
            print(f"{version}: Error reading metrics.csv - {e}")

if __name__ == "__main__":
    inspect_logs()
