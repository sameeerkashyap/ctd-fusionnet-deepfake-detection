# Lightning Classification

A basic image classification deep learning network using PyTorch Lightning with comprehensive debugging statements.

## Features

- **PyTorch Lightning Architecture**: Clean, modular code with separation of concerns
- **Mac MPS Support**: GPU acceleration on Apple Silicon (M1/M2/M3) - see [MAC_SETUP.md](MAC_SETUP.md)
- **Comprehensive Debugging**: Extensive print statements for every layer, training loop, and validation loop
- **Metrics Tracking**: Accuracy, Precision, Recall, and F1-Score for train/val/test
- **Advanced Callbacks**: ModelCheckpoint, EarlyStopping, LearningRateMonitor
- **TensorBoard Logging**: Visualize training progress and metrics
- **Flexible Data Loading**: Supports CIFAR-10 or custom datasets
- **Rich Progress Bars**: Beautiful terminal output with progress tracking

## Project Structure

```
lightning_classification/
├── model.py              # BasicImageClassifier model definition
├── datamodule.py         # ImageDataModule for data loading
├── train.py              # Training script with full debugging
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or with uv (faster)
uv pip install -r requirements.txt
```

## Quick Start for Mac Users 🍎

If you're on a Mac with Apple Silicon (M1/M2/M3):

```bash
# 1. Check if MPS (GPU) is available
python check_mps.py

# 2. Run quick test
python test_setup.py

# 3. Train with GPU acceleration (automatic)
python train.py --use_cifar10 --max_epochs 10
```

See [MAC_SETUP.md](MAC_SETUP.md) for detailed Mac-specific instructions and optimization tips.

## Usage

### Quick Start (CIFAR-10)

```bash
python train.py --use_cifar10 --max_epochs 10 --batch_size 32
```

### Custom Dataset

```bash
python train.py \
    --data_dir /path/to/your/dataset \
    --num_classes 5 \
    --max_epochs 50 \
    --batch_size 64 \
    --learning_rate 0.001
```

### All Available Arguments

```bash
# Data arguments
--data_dir              # Path to data directory (default: ./data)
--use_cifar10          # Use CIFAR-10 dataset (flag)
--num_classes          # Number of classes (default: 10)
--image_size           # Input image size (default: 224)
--train_val_split      # Train/val split ratio (default: 0.8)

# Model arguments
--learning_rate        # Learning rate (default: 1e-3)
--dropout_rate         # Dropout rate (default: 0.5)

# Training arguments
--batch_size           # Batch size (default: 32)
--max_epochs           # Maximum epochs (default: 50)
--num_workers          # Data loading workers (default: 4)
--precision            # Training precision: 32, 16, bf16 (default: 32)
--gradient_clip_val    # Gradient clipping (default: 1.0)
--accumulate_grad_batches  # Gradient accumulation (default: 1)
--val_check_interval   # Validation frequency (default: 1.0)
--early_stopping_patience  # Early stopping patience (default: 10)

# Output arguments
--output_dir           # Output directory (default: ./outputs)
--experiment_name      # Experiment name (default: experiment_1)

# Other arguments
--seed                 # Random seed (default: 42)
--run_test            # Run testing after training (flag)
```

## Model Architecture

The `BasicImageClassifier` consists of:

1. **Convolutional Blocks** (4 blocks):
   - Conv2d → BatchNorm2d → ReLU → MaxPool2d
   - Channels: 3 → 32 → 64 → 128 → 256

2. **Fully Connected Layers** (3 layers):
   - FC1: 256×14×14 → 512 (with BatchNorm + Dropout)
   - FC2: 512 → 256 (with BatchNorm + Dropout)
   - FC3: 256 → num_classes (output)

3. **Total Parameters**: ~50M (for 224×224 input)

## Debugging Features

The model includes extensive debugging at every level:

### Model Initialization
- Prints layer configurations
- Shows parameter counts
- Displays model architecture

### Forward Pass
- Layer-by-layer output shapes
- Activation statistics (min, max, mean, std)
- Dead neuron detection
- Gradient flow monitoring

### Training Loop
- Batch information (shape, label distribution)
- Loss values per batch
- Prediction vs ground truth comparison
- Metric updates in real-time

### Validation Loop
- Validation batch statistics
- Per-epoch metric summaries
- Learning rate changes
- Best model tracking

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir outputs/logs

# Open browser to http://localhost:6006
```

### Console Output

The training script provides detailed console output:
- Device information (GPU/CPU)
- Dataset statistics
- Model summary
- Training progress with metrics
- Validation results
- Best checkpoint information

## Example Output

```
============================================================
INITIALIZING MODEL
============================================================
Number of classes: 10
Input channels: 3
Learning rate: 0.001
Dropout rate: 0.5
Conv1: 3 -> 32 channels, kernel=3x3
Conv2: 32 -> 64 channels, kernel=3x3
Conv3: 64 -> 128 channels, kernel=3x3
Conv4: 128 -> 256 channels, kernel=3x3
FC1: 50176 -> 512
FC2: 512 -> 256
FC3: 256 -> 10
============================================================

############################################################
STARTING TRAINING EPOCH 0
############################################################

============================================================
TRAINING STEP - Batch 0
============================================================
Batch shape: torch.Size([32, 3, 224, 224]), Labels shape: torch.Size([32])
Label distribution: tensor([3, 4, 2, 5, 3, 4, 2, 3, 4, 2])

After Conv1: torch.Size([32, 32, 224, 224])
  Stats - min: -2.1179, max: 3.4521, mean: 0.0234
After BN1+ReLU - mean: 0.5123, std: 0.7234
  Dead neurons: 12345 / 1605632
...
```

## Tips for Debugging

1. **First Batch Debugging**: The model automatically prints detailed debug info for the first batch of each epoch
2. **Enable Full Debugging**: Pass `debug=True` to the forward pass for any batch
3. **Monitor Dead Neurons**: Check for high dead neuron counts (indicates dying ReLU problem)
4. **Watch Loss Values**: Sudden spikes or NaN values indicate training instability
5. **Check Label Distribution**: Ensure balanced batches for better training

## Customization

### Modify Model Architecture

Edit `model.py` to change:
- Number of convolutional layers
- Channel dimensions
- Fully connected layer sizes
- Activation functions

### Change Data Augmentation

Edit `datamodule.py` to modify:
- Transform pipeline
- Augmentation strategies
- Normalization values

### Adjust Training Strategy

Edit `train.py` to change:
- Optimizer (currently Adam)
- Learning rate scheduler
- Callbacks and monitoring

## Troubleshooting

### Out of Memory
- Reduce `--batch_size`
- Use `--precision 16` for mixed precision
- Increase `--accumulate_grad_batches`

### Slow Training
- Increase `--num_workers`
- Use smaller `--image_size`
- Enable GPU if available

### Poor Convergence
- Adjust `--learning_rate`
- Modify `--dropout_rate`
- Check data augmentation

## License

MIT License - Feel free to use and modify!
