#!/bin/bash

# Quick Start Script for Mac Users
# This script sets up and runs a quick training session on Mac with MPS

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Lightning Classification - Mac Quick Start                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  Warning: This script is designed for macOS"
    echo "   You appear to be running on: $OSTYPE"
    echo ""
fi

# Check Python
echo "1️⃣  Checking Python..."

# Check for virtual environment
if [ -f "../.venv/bin/python" ]; then
    PYTHON_CMD="../.venv/bin/python"
    echo "   ✓ Using virtual environment: .venv"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
    echo "   ✓ Using virtual environment: .venv"
else
    PYTHON_CMD="python"
    echo "   ✓ Using system python"
fi

if ! $PYTHON_CMD --version &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "   ✓ Python $PYTHON_VERSION found"
echo ""

# Check if dependencies are installed
echo "2️⃣  Checking dependencies..."
if $PYTHON_CMD -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$($PYTHON_CMD -c "import torch; print(torch.__version__)")
    echo "   ✓ PyTorch $TORCH_VERSION installed"
else
    echo "   ⚠️  PyTorch not found. Installing dependencies..."
    if [ "$PYTHON_CMD" == "python" ]; then
        pip install -r requirements.txt
    else
        # If using venv, install into it
        $PYTHON_CMD -m pip install -r requirements.txt
    fi
fi
echo ""

# Check MPS availability
echo "3️⃣  Checking MPS (Apple Silicon GPU) availability..."
$PYTHON_CMD check_mps.py
echo ""

# Ask user if they want to continue
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Ready to start training!                                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "This will:"
echo "  • Use local Diffusion dataset"
echo "  • Train for 10 epochs (5-10 minutes on M1/M2/M3)"
echo "  • Use your Mac's GPU automatically"
echo "  • Save results to ./outputs/"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Starting Training...                                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Run training
$PYTHON_CMD train.py \
    --max_epochs 10 \
    --batch_size 32 \
    --num_workers 2 \
    --experiment_name deepfake_quickstart

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Training Complete!                                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: ./outputs/"
echo ""
echo "To view training logs with TensorBoard:"
echo "  tensorboard --logdir outputs/logs"
echo "  Then open: http://localhost:6006"
echo ""
echo "To train for more epochs:"
echo "  $PYTHON_CMD train.py --max_epochs 50"
echo ""
