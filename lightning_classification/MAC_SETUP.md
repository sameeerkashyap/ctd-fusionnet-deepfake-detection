# Running on Mac with Apple Silicon (M1/M2/M3)

This guide helps you run the Lightning Classification project on Mac with Apple Silicon GPU acceleration using MPS (Metal Performance Shaders).

## 🍎 Mac MPS Support

PyTorch supports Apple Silicon GPUs through **MPS (Metal Performance Shaders)**. This provides significant speedups compared to CPU-only training.

### Requirements

- **Mac with Apple Silicon**: M1, M2, M3, or newer
- **macOS**: 12.3 (Monterey) or later
- **PyTorch**: 2.0.0 or later

## 🚀 Quick Start

### 1. Check MPS Availability

First, verify that your Mac supports MPS:

```bash
cd lightning_classification
python check_mps.py
```

This will:
- ✅ Check if MPS is available
- ✅ Test basic MPS operations
- ✅ Benchmark MPS vs CPU performance
- ✅ Provide recommendations

### 2. Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using uv (faster)
uv pip install -r requirements.txt
```

### 3. Run Quick Test

```bash
python test_setup.py
```

This will show which device (MPS/CPU) will be used.

### 4. Train with MPS

```bash
# Quick training (10 epochs)
python train.py --use_cifar10 --max_epochs 10 --batch_size 32

# Full training
python train.py --use_cifar10 --max_epochs 50 --batch_size 64
```

**PyTorch Lightning automatically detects and uses MPS** - no extra flags needed!

## 📊 Expected Performance

On Apple Silicon Macs, you should see:

- **M1**: ~3-5x faster than CPU
- **M2**: ~4-6x faster than CPU
- **M3**: ~5-8x faster than CPU

Actual speedup depends on:
- Model size
- Batch size
- Memory bandwidth
- Other running applications

## ⚙️ Optimization Tips for Mac

### 1. Batch Size

Start with smaller batch sizes and increase:

```bash
# Start small
python train.py --use_cifar10 --batch_size 32

# Increase if memory allows
python train.py --use_cifar10 --batch_size 64

# For M2/M3 Pro/Max with more memory
python train.py --use_cifar10 --batch_size 128
```

### 2. Number of Workers

Mac benefits from fewer workers:

```bash
# Recommended for Mac
python train.py --use_cifar10 --num_workers 2

# Or even
python train.py --use_cifar10 --num_workers 0
```

### 3. Image Size

Smaller images = faster training:

```bash
# Faster (but less accurate)
python train.py --use_cifar10 --image_size 128

# Default
python train.py --use_cifar10 --image_size 224
```

### 4. Mixed Precision

**Note**: MPS doesn't support FP16 mixed precision yet. Use default FP32:

```bash
# Use default precision on Mac
python train.py --use_cifar10 --precision 32
```

## 🐛 Troubleshooting

### MPS Not Available

If `check_mps.py` shows MPS is not available:

1. **Check macOS version**:
   ```bash
   sw_vers
   ```
   Need macOS 12.3 or later.

2. **Update PyTorch**:
   ```bash
   pip install --upgrade torch torchvision
   ```

3. **Verify Apple Silicon**:
   ```bash
   uname -m
   ```
   Should show `arm64`.

### Out of Memory

If you get memory errors:

1. **Reduce batch size**:
   ```bash
   python train.py --use_cifar10 --batch_size 16
   ```

2. **Close other apps**: Browsers, video editors, etc.

3. **Monitor memory**:
   - Open Activity Monitor
   - Check "Memory" tab
   - Look for "Memory Pressure"

4. **Use gradient accumulation**:
   ```bash
   python train.py --use_cifar10 --batch_size 16 --accumulate_grad_batches 4
   ```
   This simulates batch_size=64 with less memory.

### Slow Performance

If training is slower than expected:

1. **Check Activity Monitor**:
   - Look for other GPU-intensive apps
   - Check CPU usage (should be moderate)

2. **Reduce num_workers**:
   ```bash
   python train.py --use_cifar10 --num_workers 0
   ```

3. **Close browser tabs**: Chrome/Safari can use GPU

4. **Disable other GPU apps**: Video players, photo editors

### MPS Errors

If you encounter MPS-specific errors:

1. **Fallback to CPU**:
   ```bash
   # Force CPU usage
   PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py --use_cifar10
   ```

2. **Update PyTorch**:
   ```bash
   pip install --upgrade torch torchvision
   ```

3. **Report the issue**: Some operations may not be supported yet

## 📈 Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir outputs/logs

# Open in browser
open http://localhost:6006
```

### Activity Monitor

1. Open Activity Monitor (Applications → Utilities)
2. Click "GPU" tab
3. Watch "GPU Usage" graph
4. Monitor "Memory Pressure"

### Console Output

The training script shows:
- Device being used (MPS/CPU)
- Batch processing time
- Metrics per epoch
- Memory usage warnings

## 🎯 Recommended Settings for Mac

### M1 (8GB)
```bash
python train.py \
  --use_cifar10 \
  --batch_size 32 \
  --num_workers 2 \
  --max_epochs 50
```

### M1 Pro/Max (16GB+)
```bash
python train.py \
  --use_cifar10 \
  --batch_size 64 \
  --num_workers 4 \
  --max_epochs 50
```

### M2/M3 (8GB)
```bash
python train.py \
  --use_cifar10 \
  --batch_size 48 \
  --num_workers 2 \
  --max_epochs 50
```

### M2/M3 Pro/Max (16GB+)
```bash
python train.py \
  --use_cifar10 \
  --batch_size 128 \
  --num_workers 4 \
  --max_epochs 50
```

## 🔍 Verifying MPS Usage

During training, you should see:

```
============================================================
DEVICE INFORMATION
============================================================
CUDA available: False
MPS (Apple Silicon) available: True
MPS device: Apple Silicon GPU
MPS built: True

🚀 Accelerator to be used: MPS (Apple Silicon GPU)
PyTorch version: 2.x.x
PyTorch Lightning version: 2.x.x
============================================================
```

## 📚 Additional Resources

- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [Apple Silicon GPU Guide](https://developer.apple.com/metal/pytorch/)
- [PyTorch Lightning on Mac](https://lightning.ai/docs/pytorch/stable/)

## 🆘 Getting Help

If you encounter issues:

1. Run `python check_mps.py` and share output
2. Check PyTorch version: `python -c "import torch; print(torch.__version__)"`
3. Check macOS version: `sw_vers`
4. Share error messages from training

## ✅ Quick Checklist

Before training:
- [ ] macOS 12.3 or later
- [ ] Apple Silicon Mac (M1/M2/M3)
- [ ] PyTorch 2.0+ installed
- [ ] `python check_mps.py` shows MPS available
- [ ] Closed other GPU-intensive apps
- [ ] Checked available memory

Ready to train! 🚀

---

**Note**: MPS support in PyTorch is continuously improving. Some operations may fall back to CPU, which is normal and handled automatically.
