import os
import torch
import cv2
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import glob
import random
from flask import Flask, request, jsonify, render_template
from lightning_impl.lightning_module import FusionNetLightning
from lightning_impl.dataset import fast_ctd_residual
from lightning_impl.transforms import get_transforms

app = Flask(__name__)

# Configuration
CHECKPOINT_PATH = "checkpoints-inverted/fusionnet-epoch=08-val_auc=1.0000.ckpt"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
MODEL = None
EXPLAINER = None
BACKGROUND_DATA = None

def load_model():
    global MODEL
    if MODEL is None:
        print(f"Loading model from {CHECKPOINT_PATH}...")
        try:
            MODEL = FusionNetLightning.load_from_checkpoint(
                CHECKPOINT_PATH,
                map_location=DEVICE
            )
            MODEL.to(DEVICE)
            MODEL.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

def load_background_data(n_samples=1):
    """Load a single zero image to serve as background for SHAP (Fastest)."""
    global BACKGROUND_DATA
    if BACKGROUND_DATA is not None:
        return BACKGROUND_DATA

    print("Loading background data for SHAP (using zero baseline for speed)...")
    # Use a single zero image as background. 
    # This is much faster than using real images and often sufficient for "what features are present vs absent".
    dummy_img = torch.zeros(1, 3, 224, 224).to(DEVICE)
    dummy_noise = torch.zeros(1, 3, 224, 224).to(DEVICE)
    BACKGROUND_DATA = [dummy_img, dummy_noise]
    
    return BACKGROUND_DATA

def preprocess_image(image_file):
    # Read image from file object
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise ValueError("Could not decode image")

    # Convert to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Compute CTD noise
    noise = fast_ctd_residual(img_rgb)
    
    # Get transforms (val/test mode)
    transforms = get_transforms(is_train=False)
    
    # Apply transforms
    transformed = transforms(image=img_rgb, noise=noise)
    
    img_tensor = transformed['image'].unsqueeze(0).to(DEVICE)
    noise_tensor = transformed['noise'].unsqueeze(0).to(DEVICE)
    
    return img_tensor, noise_tensor

def get_shap_explanation(img_tensor, noise_tensor):
    global EXPLAINER
    
    print("Generating SHAP explanation...")
    
    # Ensure background data is loaded
    bg_data = load_background_data()
    
    if EXPLAINER is None:
        print("Initializing GradientExplainer...")
        # GradientExplainer expects the model and background data
        # The model forward pass takes (img, noise)
        EXPLAINER = shap.GradientExplainer(MODEL, bg_data)

    # Compute shap values
    print("Computing SHAP values (this might take a while)...")
    # shap_values is a list of arrays (one for each class)
    # Each element is a list of tensors corresponding to inputs [img, noise]
    shap_values = EXPLAINER.shap_values([img_tensor, noise_tensor])
    print("SHAP values computed.")
    
    # Inverse normalize for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img_np = img_tensor.cpu().detach().numpy()[0].transpose(1, 2, 0)
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    
    # SHAP values for RGB input (index 0 of input list)
    # We'll focus on RGB explanation for visual clarity
    # shap_values is list of [shap_img, shap_noise] for each class
    
    # Let's pick the class we are interested in. 
    # For binary, usually class 1 (Real) vs class 0 (Fake).
    # Let's show explanation for Class 0 (Fake) to see what makes it fake.
    
    shap_val_rgb = shap_values[0][0] # Class 0, Input 0 (RGB)
    # shap_val_rgb is (1, 3, 224, 224) -> need (224, 224, 3)
    shap_val_rgb_np = shap_val_rgb[0].transpose(1, 2, 0)
    
    # Plot
    print("Generating SHAP plot...")
    plt.figure()
    # shap.image_plot expects numpy arrays with batch dimension (N, H, W, C)
    pixel_values = np.array([img_np]) # (1, H, W, C)
    shap_values_plot = np.array([shap_val_rgb_np]) # (1, H, W, C)
    
    shap.image_plot(shap_values_plot, pixel_values, show=False)
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    print("SHAP plot generated.")
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    try:
        print(f"Received file for prediction: {file.filename}")
        img_tensor, noise_tensor = preprocess_image(file)
        
        print("Running inference...")
        with torch.no_grad():
            logits = MODEL(img_tensor, noise_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, 1)
            
            # Model classes are inverted: 0=Real, 1=Fake
            # So if pred == 0 -> Real
            # If pred == 1 -> Fake
            prediction = "REAL" if pred.item() == 0 else "FAKE"
            confidence = conf.item()
            
            # Get probabilities for both classes
            # Class 0 is Real, Class 1 is Fake
            real_prob = probs[0][0].item()
            fake_prob = probs[0][1].item()
        
        print(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
            
        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'fake': fake_prob,
                'real': real_prob
            }
        })
        
    except Exception as e:
        print(f"Error processing prediction request: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/explain', methods=['POST'])
def explain():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    try:
        print(f"Received file for explanation: {file.filename}")
        img_tensor, noise_tensor = preprocess_image(file)
        
        # Enable gradients for SHAP
        img_tensor.requires_grad = True
        noise_tensor.requires_grad = True
        
        # Generate SHAP explanation
        shap_image = get_shap_explanation(img_tensor, noise_tensor)
            
        return jsonify({
            'shap_image': shap_image
        })
        
    except Exception as e:
        print(f"Error processing explanation request: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    # Pre-load background data to save time on first request
    load_background_data()
    app.run(debug=True, port=5001)
