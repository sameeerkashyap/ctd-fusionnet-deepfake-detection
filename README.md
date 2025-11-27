# CTD-FusionNet: Advanced Deepfake Detection

A state-of-the-art deepfake detection system using Copy-Move Tampering Detection (CTD) with multi-modal fusion of RGB and noise features.

## 🏆 Key Features

- **Multi-Modal Architecture**: Combines RGB images with CTD residual noise for robust detection
- **Advanced Fusion**: ConvNeXt + EfficientNet-B0 + Swin Transformer with attention-based fusion
- **Modular Design**: Clean separation of dataset, model, training, and evaluation components
- **Configurable Training**: Extensive configuration options for experimentation
- **Debug Mode**: Ultra-fast debugging with dataset subsampling
- **Progress Monitoring**: Real-time training progress with tqdm bars
- **Comprehensive Metrics**: AUC, Accuracy, Precision, Recall, F1-score tracking

## 📊 Performance

- **Architecture**: 60M+ parameters across 3 backbone networks
- **Input**: RGB images + CTD noise maps
- **Output**: Binary classification (Real/Fake)
- **Training**: Optimized for large-scale datasets

## 🏗️ Project Structure

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

## 🚀 Quick Start

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
   source .venv/bin/activate
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

## 🎯 Usage

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
    'loss_type': 'cross_entropy',  # 'cross_entropy' (only option)

    # Augmentations (uncomment to enable)
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

## 🧠 Model Architecture

### CTD-FusionNet Components

1. **RGB Branch**: ConvNeXt-Tiny (27.8M params)
   - Processes RGB input images
   - Extracts high-level visual features

2. **Noise Branch**: EfficientNet-B0 (3.6M params)
   - Processes CTD residual noise
   - Captures tampering artifacts

3. **Spatial Branch**: Swin Transformer (27.5M params)
   - Processes resized RGB images
   - Captures spatial relationships

4. **Attention Fusion**: Custom attention mechanism (0.9M params)
   - Fuses RGB and noise features
   - Uses batched matrix multiplication for stability

### CTD Noise Generation

The system automatically generates Copy-Move Tampering Detection noise:

```python
def fast_ctd_residual(img_rgb_uint8):
    den = cv2.GaussianBlur(img_rgb_uint8, (5, 5), 0)
    noise = np.clip(img_rgb_uint8 - den + 128, 0, 255).astype(np.uint8)
    return noise
```

## 🔧 Development

### Debug Mode Features

- **Dataset subsampling**: Uses ~50 train, ~25 val, ~25 test samples
- **Augmentation control**: Automatically skips augmentations when only 'normalize' is specified
- **Fast iteration**: Complete training runs in seconds

### Adding Custom Components

- **Models**: Add to `model.py` following the existing pattern
- **Transforms**: Modify `transforms.py` for custom augmentations
- **Datasets**: Extend `dataset.py` for different data sources
- **Training**: Update `train.py` config system for new options

## 📈 Training Monitoring

The system provides comprehensive progress tracking:

```
Training:  69%|██████▉   | 9/13 [loss=0.5127, avg_loss=0.5961]
Validating: 100%|██████████| 6/6 [00:01<00:00, 5.32batch/s]
Testing: 100%|██████████| 1/1 [00:00<00:00, 7.14batch/s]
```

## 🤝 Contributing

1. Follow the modular architecture pattern
2. Use the config system for new parameters
3. Add tqdm progress bars for long operations
4. Test with debug mode before full training
5. Update documentation for new features

## 📄 License

See LICENSE file for details.

## ⚠️ Notes

- **Debug mode** gives artificially high performance due to overfitting on small datasets
- **Use full datasets** for meaningful performance evaluation
- **GPU recommended** for training (CPU debug mode available)
- **Pretrained weights** are downloaded automatically for backbone networks
