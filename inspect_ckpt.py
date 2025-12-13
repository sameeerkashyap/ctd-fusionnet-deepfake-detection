import torch

def inspect_checkpoint():
    ckpt_path = "checkpoints-inverted/fusionnet-epoch=08-val_auc=1.0000.ckpt"
    try:
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        print(f"Checkpoint keys: {checkpoint.keys()}")
        
        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")
            
        if 'global_step' in checkpoint:
            print(f"Global Step: {checkpoint['global_step']}")
            
        if 'callbacks' in checkpoint:
            print(f"Callbacks: {checkpoint['callbacks'].keys()}")
            
        # Check if there is any history in loops or other keys
        # Usually not, but worth checking
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    inspect_checkpoint()
