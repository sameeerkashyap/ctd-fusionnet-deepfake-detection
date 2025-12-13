"""
Inference script for CTD-FusionNet using PyTorch Lightning checkpoints.

Usage:
    python lightning_impl/inference.py --checkpoint checkpoints/best_model.ckpt --input path/to/image.jpg
    python lightning_impl/inference.py --checkpoint checkpoints/best_model.ckpt --input path/to/folder/
"""

import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

# Import model and utils
# Ensure we can import from the current directory
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lightning_module import FusionNetLightning
from transforms import get_transforms
from dataset import fast_ctd_residual

def load_model(checkpoint_path, device=None):
    """
    Load the trained model from a checkpoint.
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
    print(f"🚀 Loading model from {checkpoint_path}")
    print(f"   Device: {device}")

    # Load model from checkpoint
    # strict=False allows loading even if some keys are missing (though they shouldn't be)
    model = FusionNetLightning.load_from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    
    return model, device

def preprocess_image(image_path, transform):
    """
    Read and preprocess an image for the model.
    """
    # Read image using OpenCV (to match training pipeline)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    # Convert to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Generate CTD noise
    noise = fast_ctd_residual(img_rgb)
    
    # Apply transforms
    # We use validation transforms (no augmentation)
    transformed = transform(image=img_rgb, noise=noise)
    
    return transformed['image'], transformed['noise']

def predict(model, device, image_tensor, noise_tensor):
    """
    Run inference on a single image.
    """
    # Add batch dimension
    img_batch = image_tensor.unsqueeze(0).to(device)
    noise_batch = noise_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(img_batch, noise_batch)
        probs = torch.softmax(logits, dim=1)
        
        # Get probability of Fake (class 1)
        fake_prob = probs[0, 1].item()
        
    return fake_prob

def main():
    parser = argparse.ArgumentParser(description="Run inference with CTD-FusionNet")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt model file")
    parser.add_argument("--input", type=str, required=True, help="Path to image file or directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (default: 0.5)")
    args = parser.parse_args()

    # Setup
    model, device = load_model(args.checkpoint)
    
    # Configuration used during training
    # MUST match lightning_impl/train.py
    inference_config = {
        'normalize': {
            'mean': (0.433168, 0.375803, 0.337967),
            'std': (0.270012, 0.247510, 0.239540)
        }
    }

    # Get transforms (validation mode) with correct normalization
    transform = get_transforms(
        img_size=224, 
        augmentation_config=inference_config,
        is_train=False
    )
    
    # Process input
    if os.path.isfile(args.input):
        image_paths = [args.input]
    elif os.path.isdir(args.input):
        image_paths = [
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))
        ]
        image_paths.sort()
    else:
        print(f"❌ Error: Input {args.input} not found.")
        return

    print(f"\n🔍 Running inference on {len(image_paths)} images...")
    print("-" * 60)
    print(f"{'Image Name':<40} | {'Prediction':<10} | {'Confidence':<10}")
    print("-" * 60)

    results = []
    
    for img_path in tqdm(image_paths, disable=len(image_paths) < 5):
        try:
            img_tensor, noise_tensor = preprocess_image(img_path, transform)
            fake_prob = predict(model, device, img_tensor, noise_tensor)
            
            label = "FAKE" if fake_prob >= args.threshold else "REAL"
            confidence = fake_prob if label == "FAKE" else 1 - fake_prob
            
            # Print result for single images or small batches
            if len(image_paths) <= 20:
                print(f"{os.path.basename(img_path):<40} | {label:<10} | {confidence:.4f}")
            
            results.append({
                'path': img_path,
                'prob': fake_prob,
                'label': label
            })
            
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(img_path)}: {e}")

    # Summary
    if len(image_paths) > 1:
        fake_count = sum(1 for r in results if r['label'] == 'FAKE')
        real_count = len(results) - fake_count
        print("-" * 60)
        print(f"📊 Summary:")
        print(f"   Total Images: {len(results)}")
        print(f"   Detected FAKE: {fake_count}")
        print(f"   Detected REAL: {real_count}")

if __name__ == "__main__":
    main()
