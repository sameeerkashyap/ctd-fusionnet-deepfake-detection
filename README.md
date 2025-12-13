# CTD-FusionNet: Advanced Deepfake Detection

A state-of-the-art deepfake detection system using Copy-Move Tampering Detection (CTD) with multi-modal fusion of RGB and noise features.

## Key Features

* **Multi-Modal Architecture**: Combines RGB images with CTD residual noise for robust detection
* **Advanced Fusion**: ConvNeXt + EfficientNet-B0 + Swin Transformer with attention-based fusion
* **Modular Design**: Clean separation of dataset, model, training, and evaluation components
* **Configurable Training**: Extensive configuration options for experimentation
* **Debug Mode**: Ultra-fast debugging with dataset subsampling
* **Progress Monitoring**: Real-time training progress with tqdm bars
* **Comprehensive Metrics**: AUC, Accuracy, Precision, Recall, F1-score tracking

## Performance

* **Architecture**: 60M+ parameters across 3 backbone networks
* **Input**: RGB images + CTD noise maps
* **Output**: Binary classification (Real/Fake)
* **Training**: Optimized for large-scale datasets

## Project Structure

```
ctd-fusionnet-deepfake-detection/
├── model.py              # CTD-FusionNet architecture
├── dataset.py            # DeepfakeDataset with CTD noise generation
├── transforms.py         # Albumentations-based augmentations
├── train.py              # Training pipeline with config system
├── calculate_stats.py    # Dataset statistics calculator
├── main.ipynb            # Original development notebook
├── pyproject.toml        # Dependencies
├── uv.lock              # Lock file
├── checkpoints/         # Saved models (created during training)
└── infrastructure/      # Terraform deployment configs
```

# Try the app


Download the checkpoints from here:
and place the folder in the root of the file from [here](https://drive.google.com/drive/folders/1PUWKOy6uTWFQB6Mvdg1_grOjlNRwBDjq?usp=sharing)

Run the app:
```bash
uv run app.py
```

**Testing**
- Use the images in test_app/fake to test the app with fake images
- Use the images in test_app/real to test the app with real images

## Quick Start

### Environment Setup (uv)

1. **Install uv** (if not already available):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Create environment and install dependencies**:
```bash
uv sync --python 3.10
```

3. **Activate environment**:
```bash
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

4. **(Optional) Register Jupyter kernel**:
```bash
.venv/bin/python -m ipykernel install --user --name ctd-fusionnet --display-name "ctd-fusionnet (uv)"
```

### Dataset Preparation

1. **Place your dataset** in `../Dataset/` with this structure:
```
Dataset/
├── Train/
│   ├── Real/     # Real training images
│   └── Fake/     # Fake training images
├── Val/
│   ├── Real/     # Real validation images
│   └── Fake/     # Fake validation images
└── Test/
    ├── Real/     # Real test images
    └── Fake/     # Fake test images
```

2. **Calculate dataset statistics** (optional, for custom normalization):
```bash
python calculate_stats.py --max-samples 5000
```

## Usage

### Training

**Full training** (recommended for production):
```bash
python train.py
```

**Debug training** (fast iteration, small dataset):
```python
from train import CONFIG, train_model
CONFIG['debug'] = True
CONFIG['num_epochs'] = 2
checkpoint_path = train_model(CONFIG)
```

### Testing

**Test a trained model**:
```python
from train import test_checkpoint
metrics = test_checkpoint('checkpoints/best_model.pth')
print(f"Test AUC: {metrics['auc']:.4f}")
```

### Configuration

The system uses a centralized config system. Key options:

```python
CONFIG = {
    # Dataset
    'dataset_root': "../Dataset",
    'img_size': 224,

    # Training
    'batch_size': 32,
    'learning_rate': 1e-4,
    'num_epochs': 10,
    'weight_decay': 1e-4,

    # Loss function
    'loss_type': 'cross_entropy',

    # Augmentations
    'augmentation': {
        # 'crop_scale': (0.75, 1.0),
        # 'horizontal_flip_p': 0.5,
        'normalize': {
            'mean': (0.433168, 0.375803, 0.337967),
            'std': (0.270012, 0.247510, 0.239540)
        }
    },

    # Debug mode
    'debug': False,  # Set True for fast debugging
}
```

## Model Architecture

### CTD-FusionNet Components

1. **RGB Branch**: ConvNeXt-Tiny (27.8M params)
   * Processes RGB input images
   * Extracts high-level visual features

2. **Noise Branch**: EfficientNet-B0 (3.6M params)
   * Processes CTD residual noise
   * Captures tampering artifacts

3. **Spatial Branch**: Swin Transformer (27.5M params)
   * Processes resized RGB images
   * Captures spatial relationships

4. **Attention Fusion**: Custom attention mechanism (0.9M params)
   * Fuses RGB and noise features
   * Uses batched matrix multiplication for stability

### CTD Noise Generation

The system automatically generates Copy-Move Tampering Detection noise:

```python
def fast_ctd_residual(img_rgb_uint8):
    den = cv2.GaussianBlur(img_rgb_uint8, (5, 5), 0)
    noise = np.clip(img_rgb_uint8 - den + 128, 0, 255).astype(np.uint8)
    return noise
```

**Explanation**: This function creates a noise map by subtracting a Gaussian-blurred (denoised) version of the image from the original. The result reveals high-frequency artifacts and compression noise that are characteristic of deepfakes. The offset of 128 centers the residual values around mid-gray for better visualization.

## Development

### Debug Mode Features

* **Dataset subsampling**: Uses ~50 train, ~25 val, ~25 test samples
* **Augmentation control**: Automatically skips augmentations when only 'normalize' is specified
* **Fast iteration**: Complete training runs in seconds

### Adding Custom Components

* **Models**: Add to `model.py` following the existing pattern
* **Transforms**: Modify `transforms.py` for custom augmentations
* **Datasets**: Extend `dataset.py` for different data sources
* **Training**: Update `train.py` config system for new options

## Code Structure Explained

### 1. model.py - Neural Network Architecture

The core of the system is the CTD-FusionNet model, which implements a multi-branch architecture:

**RGB Branch (ConvNeXt-Tiny)**:
- Takes standard RGB images as input (224x224x3)
- ConvNeXt is a modernized ConvNet that achieves competitive performance with transformers
- Extracts semantic features like faces, textures, and visual patterns
- Outputs a 768-dimensional feature vector

**Noise Branch (EfficientNet-B0)**:
- Takes CTD residual noise maps as input (224x224x3)
- EfficientNet-B0 is a lightweight, efficient CNN
- Specializes in detecting compression artifacts and tampering traces
- Outputs a 1280-dimensional feature vector

**Spatial Branch (Swin Transformer)**:
- Takes RGB images as input but processes them differently
- Swin Transformer uses shifted windows for efficient attention computation
- Captures long-range spatial dependencies and context
- Outputs a 768-dimensional feature vector

**Attention Fusion Module**:
- Combines features from all three branches
- Uses learnable attention weights to determine importance of each branch
- Implements cross-attention between RGB and noise features
- Final classification head outputs probability of fake vs real

The model architecture can be summarized as:
```
Input Image -> [RGB Branch] -> Features_RGB (768-dim)
            -> [CTD Noise] -> [Noise Branch] -> Features_Noise (1280-dim)
            -> [Spatial Branch] -> Features_Spatial (768-dim)
            -> [Attention Fusion] -> [Classifier] -> Prediction (fake/real)
```

### 2. dataset.py - Data Loading and Preprocessing

This module handles loading images and generating CTD noise:

**DeepfakeDataset Class**:
- Inherits from PyTorch's Dataset class
- Automatically discovers images in Train/Val/Test folders
- Labels: Real=0, Fake=1
- Returns tuples of (rgb_image, noise_image, label)

**CTD Noise Generation Pipeline**:
1. Load RGB image using OpenCV
2. Apply Gaussian blur with 5x5 kernel to create denoised version
3. Compute residual: original - denoised + 128
4. Clip values to valid range [0, 255]
5. Return as uint8 numpy array

**Key Features**:
- Efficient caching of file paths
- Support for common image formats (jpg, png, jpeg)
- Automatic label assignment based on folder structure
- Compatible with PyTorch DataLoader for batching

### 3. transforms.py - Data Augmentation

Uses Albumentations library for advanced augmentations:

**Training Augmentations** (optional, configured in CONFIG):
- **RandomResizedCrop**: Randomly crops and resizes to introduce scale variation
- **HorizontalFlip**: Mirrors images horizontally to increase diversity
- **Normalization**: Standardizes pixel values using dataset statistics

**Validation/Test Transforms**:
- Only applies normalization (no augmentation to preserve test integrity)
- Uses pre-computed mean and std from calculate_stats.py

**Why Albumentations?**
- Faster than torchvision transforms
- Supports simultaneous transformation of multiple images (RGB + noise)
- Rich set of augmentations specifically designed for computer vision

### 4. train.py - Training Pipeline

The main training orchestration script with several key functions:

**CONFIG Dictionary**:
- Central configuration system for all hyperparameters
- Includes dataset paths, model settings, training parameters
- Debug mode for rapid prototyping with small data samples

**train_model() Function**:
- Creates DataLoaders for train/val/test splits
- Initializes CTD-FusionNet model
- Sets up optimizer (AdamW) and loss function (CrossEntropyLoss)
- Training loop with:
  - Forward pass through model
  - Loss computation
  - Backward propagation
  - Parameter updates
  - Validation after each epoch
  - Checkpointing best model based on validation AUC

**Evaluation Metrics**:
- **AUC (Area Under ROC Curve)**: Measures model's ability to discriminate
- **Accuracy**: Overall correctness
- **Precision**: Of predicted fakes, how many are actually fake
- **Recall**: Of actual fakes, how many are detected
- **F1-Score**: Harmonic mean of precision and recall

**test_checkpoint() Function**:
- Loads saved model from checkpoint
- Evaluates on test set
- Returns comprehensive metrics

### 5. calculate_stats.py - Dataset Statistics

Computes normalization parameters for the dataset:

**Purpose**:
- Calculates mean and standard deviation of pixel values
- These statistics are used to normalize images during training
- Normalization helps models converge faster and perform better

**Process**:
1. Samples random images from training set
2. Computes per-channel mean (R, G, B)
3. Computes per-channel standard deviation
4. Outputs values to use in CONFIG['augmentation']['normalize']

**Why Pre-compute Stats?**
- Computing statistics on-the-fly is expensive
- Different datasets have different distributions
- Custom stats improve performance vs. ImageNet defaults

## Training Monitoring

The system provides comprehensive progress tracking:

```
Training:  69%|██████▉   | 9/13 [loss=0.5127, avg_loss=0.5961]
Validating: 100%|██████████| 6/6 [00:01<00:00, 5.32batch/s]
Testing: 100%|██████████| 1/1 [00:00<00:00, 7.14batch/s]
```

These progress bars show:
- Current batch/total batches
- Processing speed (batches per second)
- Current loss values during training
- Estimated time remaining

## Contributing

1. Follow the modular architecture pattern
2. Use the config system for new parameters
3. Add tqdm progress bars for long operations
4. Test with debug mode before full training
5. Update documentation for new features

## License

Apache-2.0 License

## Important Notes

* **Debug mode** gives artificially high performance due to overfitting on small datasets
* **Use full datasets** for meaningful performance evaluation
* **GPU recommended** for training (CPU debug mode available)
* **Pretrained weights** are downloaded automatically for backbone networks

## Technical Implementation Details

### Why Multi-Modal Fusion?

Deepfakes are created by neural networks that learn to generate realistic-looking faces. However, they leave behind subtle traces:

1. **Visual Artifacts**: Blurriness, unnatural lighting, facial inconsistencies
2. **Noise Patterns**: Compression artifacts, GAN-specific noise signatures
3. **Spatial Inconsistencies**: Unnatural face boundaries, warping artifacts

The CTD-FusionNet addresses all three by using:
- RGB branch for visual semantics
- Noise branch for compression/generation artifacts
- Spatial branch for contextual relationships

### Copy-Move Tampering Detection (CTD)

CTD was originally designed to detect copy-paste forgeries but works excellently for deepfakes:

**How it Works**:
- Real images have consistent noise across the entire image
- Deepfakes have inconsistent noise where faces are generated
- The residual (original - denoised) reveals these inconsistencies

**Mathematical Formulation**:
```
Denoised = GaussianBlur(Original, kernel_size=5)
Residual = Original - Denoised + 128
CTD_Noise = clip(Residual, 0, 255)
```

The Gaussian blur acts as a low-pass filter, removing high-frequency details. Subtracting it isolates the high-frequency noise component.

### Attention Mechanism

The attention fusion module learns to weight features dynamically:

**Benefits**:
- Different samples may require different feature emphasis
- RGB features important for well-lit images
- Noise features critical for compressed/low-quality images
- Adaptive weighting improves robustness

**Implementation**:
- Learnable query, key, value projections
- Softmax-based attention weights
- Residual connections for gradient flow
- Layer normalization for training stability

### Training Strategy

**Loss Function**: Cross-Entropy Loss
- Standard for binary classification
- Penalizes confident wrong predictions heavily
- Well-suited for balanced datasets

**Optimizer**: AdamW (Adam with Weight Decay)
- Adaptive learning rates per parameter
- Weight decay acts as L2 regularization
- Prevents overfitting to training data

**Learning Rate**: 1e-4
- Conservative rate suitable for fine-tuning pretrained models
- Prevents catastrophic forgetting of pretrained features

**Batch Size**: 32
- Balance between gradient stability and memory usage
- Larger batches provide more stable gradients
- Constrained by GPU memory (60M+ parameters)

### Debug Mode Implementation

Debug mode is essential for rapid development:

**Dataset Subsampling**:
- Randomly samples ~50 train, ~25 val, ~25 test images
- Maintains class balance (equal real/fake)
- Allows full training loop testing in seconds

**Augmentation Skipping**:
- Only applies normalization in debug mode
- Reduces computational overhead
- Faster iteration cycles

**Use Cases**:
- Testing new model architectures
- Debugging training pipeline
- Validating data loading
- Quick experiments with hyperparameters

### Performance Considerations

**Model Size**: 60M+ parameters
- RGB: 27.8M (ConvNeXt-Tiny)
- Noise: 3.6M (EfficientNet-B0)
- Spatial: 27.5M (Swin Transformer)
- Fusion: 0.9M (Attention + Classifier)

**Memory Requirements**:
- Training: ~8-12GB GPU memory (batch_size=32)
- Inference: ~2-4GB GPU memory
- CPU mode available but significantly slower

**Speed**:
- Training: ~100-200 samples/second (GPU)
- Inference: ~50-100 samples/second (GPU)
- Depends on hardware, image resolution, batch size

## Repository

Original repository: [https://github.com/sameeerkashyap/ctd-fusionnet-deepfake-detection](https://github.com/sameeerkashyap/ctd-fusionnet-deepfake-detection)

## Authors

**Abhinav Sudhakar Dubey**  
Department of Computer Science  
University of California Santa Cruz  
Santa Cruz, CA 95064  
adubey4@ucsc.edu

**Sameera Sudarshan Kashyap**  
Department of Computer Science  
University of California Santa Cruz  
Santa Cruz, CA 95064  
ssudars1@ucsc.edu

**Serene Cheng**  
Department of Computer Science  
University of California Santa Cruz  
Santa Cruz, CA 95064  
scheng43@ucsc.edu

