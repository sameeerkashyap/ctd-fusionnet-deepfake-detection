#!/usr/bin/env python3
"""
Inference script for Deepfake Classification
Automatically loads the latest checkpoint and classifies a given image.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
from pathlib import Path
import sys
import os

# Add current directory to path to allow importing model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import DeepfakeClassifier

def get_latest_checkpoint(checkpoint_dir="./outputs/checkpoints"):
    """Find the latest .ckpt file in the directory"""
    p = Path(checkpoint_dir)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    checkpoints = list(p.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No .ckpt files found in {checkpoint_dir}")
    
    # Sort by modification time (newest first)
    latest_ckpt = max(checkpoints, key=lambda x: x.stat().st_mtime)
    return latest_ckpt

def load_model(checkpoint_path):
    """Load the model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load model from checkpoint
    # strict=False allows loading even if there are minor mismatches (though shouldn't be for same code)
    model = DeepfakeClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Move to appropriate device
    if torch.cuda.is_available():
        model.cuda()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        model.to(torch.device("mps"))
    
    return model

def preprocess_image(image_path, image_size=224):
    """Load and preprocess image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Define transforms (must match training validation transforms)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load image
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Failed to open image: {e}")
    
    # Apply transforms and add batch dimension
    img_tensor = transform(img).unsqueeze(0)
    
    return img_tensor

def predict(model, img_tensor):
    """Run inference"""
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        
        # Get prediction
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
        
        # Get all probabilities
        fake_prob = probs[0][0].item()
        real_prob = probs[0][1].item()
        
    return pred_idx, confidence, fake_prob, real_prob

def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection Inference")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to specific checkpoint (optional, defaults to latest)")
    parser.add_argument("--checkpoint_dir", type=str, default="./outputs/checkpoints",
                        help="Directory containing checkpoints")
    
    args = parser.parse_args()
    
    try:
        # 1. Get checkpoint
        if args.checkpoint:
            ckpt_path = Path(args.checkpoint)
            if not ckpt_path.exists():
                print(f"❌ Error: Checkpoint not found at {ckpt_path}")
                return
        else:
            try:
                ckpt_path = get_latest_checkpoint(args.checkpoint_dir)
            except FileNotFoundError as e:
                print(f"❌ Error: {e}")
                return
            
        # 2. Load model
        model = load_model(ckpt_path)
        
        # 3. Preprocess image
        img_tensor = preprocess_image(args.image_path)
        
        # 4. Run prediction
        pred_idx, confidence, fake_prob, real_prob = predict(model, img_tensor)
        
        # 5. Output results
        # Assuming class 0 is Fake and class 1 is Real (standard ImageFolder sorting)
        class_names = ['Fake', 'Real']
        prediction = class_names[pred_idx]
        
        print("\n" + "="*50)
        print(f"🖼️  Image: {args.image_path}")
        print("="*50)
        
        # Color output based on prediction
        if prediction == "Fake":
            color_code = "\033[91m" # Red
        else:
            color_code = "\033[92m" # Green
        reset_code = "\033[0m"
        
        print(f"Prediction: {color_code}{prediction.upper()}{reset_code}")
        print(f"Confidence: {confidence:.2%}")
        print("-" * 30)
        print(f"Real Probability: {real_prob:.4f}")
        print(f"Fake Probability: {fake_prob:.4f}")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
