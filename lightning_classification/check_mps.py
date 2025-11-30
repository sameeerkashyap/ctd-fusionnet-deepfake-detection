#!/usr/bin/env python3
"""
Mac MPS (Apple Silicon) Configuration and Testing Script

This script helps configure and test PyTorch with MPS (Metal Performance Shaders)
for Apple Silicon Macs (M1, M2, M3, etc.)
"""

import torch
import sys


def check_mps_availability():
    """Check if MPS is available and properly configured"""
    print("\n" + "="*70)
    print("MAC MPS (APPLE SILICON) CONFIGURATION CHECK")
    print("="*70)
    
    # System info
    print(f"\nPython version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check MPS availability
    print("\n" + "-"*70)
    print("MPS AVAILABILITY CHECK")
    print("-"*70)
    
    has_mps_attr = hasattr(torch.backends, 'mps')
    print(f"✓ torch.backends.mps exists: {has_mps_attr}")
    
    if has_mps_attr:
        is_built = torch.backends.mps.is_built()
        print(f"✓ MPS built: {is_built}")
        
        is_available = torch.backends.mps.is_available()
        print(f"✓ MPS available: {is_available}")
        
        if is_available:
            print("\n✅ MPS IS READY TO USE!")
            print("   Your Mac's GPU will be used for training.")
            return True
        else:
            print("\n⚠️  MPS is built but not available.")
            print("   Possible reasons:")
            print("   - Not running on Apple Silicon (M1/M2/M3)")
            print("   - macOS version too old (need macOS 12.3+)")
            return False
    else:
        print("\n❌ MPS NOT AVAILABLE")
        print("   Your PyTorch installation doesn't support MPS.")
        print("   You may need to update PyTorch:")
        print("   pip install --upgrade torch torchvision")
        return False


def test_mps_operations():
    """Test basic MPS operations"""
    print("\n" + "-"*70)
    print("MPS OPERATIONS TEST")
    print("-"*70)
    
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("⚠️  Skipping MPS tests (not available)")
        return False
    
    try:
        # Create tensors on MPS
        print("\n1. Creating tensor on MPS device...")
        device = torch.device("mps")
        x = torch.randn(100, 100, device=device)
        print(f"   ✓ Created tensor: {x.shape} on {x.device}")
        
        # Matrix multiplication
        print("\n2. Testing matrix multiplication...")
        y = torch.randn(100, 100, device=device)
        z = torch.mm(x, y)
        print(f"   ✓ Matrix multiplication successful: {z.shape}")
        
        # Neural network layer
        print("\n3. Testing neural network layer...")
        layer = torch.nn.Linear(100, 50).to(device)
        output = layer(x)
        print(f"   ✓ Linear layer successful: {output.shape}")
        
        # Backward pass
        print("\n4. Testing backward pass...")
        loss = output.sum()
        loss.backward()
        print(f"   ✓ Backward pass successful")
        
        # Move to CPU
        print("\n5. Testing CPU transfer...")
        cpu_tensor = z.cpu()
        print(f"   ✓ Transfer to CPU successful: {cpu_tensor.device}")
        
        print("\n✅ ALL MPS OPERATIONS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ MPS TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_devices():
    """Benchmark MPS vs CPU performance"""
    print("\n" + "-"*70)
    print("PERFORMANCE BENCHMARK")
    print("-"*70)
    
    import time
    
    # Test parameters
    size = 1000
    iterations = 100
    
    # CPU benchmark
    print(f"\nBenchmarking CPU (matrix multiply {size}x{size}, {iterations} iterations)...")
    cpu_device = torch.device("cpu")
    x_cpu = torch.randn(size, size, device=cpu_device)
    y_cpu = torch.randn(size, size, device=cpu_device)
    
    start = time.time()
    for _ in range(iterations):
        z_cpu = torch.mm(x_cpu, y_cpu)
    cpu_time = time.time() - start
    print(f"CPU time: {cpu_time:.4f} seconds")
    
    # MPS benchmark (if available)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"\nBenchmarking MPS (matrix multiply {size}x{size}, {iterations} iterations)...")
        mps_device = torch.device("mps")
        x_mps = torch.randn(size, size, device=mps_device)
        y_mps = torch.randn(size, size, device=mps_device)
        
        # Warm up
        for _ in range(10):
            _ = torch.mm(x_mps, y_mps)
        
        start = time.time()
        for _ in range(iterations):
            z_mps = torch.mm(x_mps, y_mps)
        mps_time = time.time() - start
        print(f"MPS time: {mps_time:.4f} seconds")
        
        speedup = cpu_time / mps_time
        print(f"\n🚀 MPS Speedup: {speedup:.2f}x faster than CPU")
    else:
        print("\n⚠️  MPS not available for benchmarking")


def print_recommendations():
    """Print recommendations for Mac users"""
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR MAC USERS")
    print("="*70)
    
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    if mps_available:
        print("\n✅ Your Mac is ready for GPU-accelerated training!")
        print("\nTo use MPS with the training script:")
        print("  python train.py --use_cifar10 --max_epochs 10")
        print("\nPyTorch Lightning will automatically use MPS (no extra flags needed)")
        print("\nFor better performance:")
        print("  - Use batch sizes that are multiples of 8 (e.g., 32, 64, 128)")
        print("  - Start with smaller models and scale up")
        print("  - Monitor memory usage with Activity Monitor")
        print("  - Close other GPU-intensive apps (browsers, video editors)")
    else:
        print("\n⚠️  MPS not available on your system")
        print("\nOptions:")
        print("  1. Update PyTorch: pip install --upgrade torch torchvision")
        print("  2. Check macOS version (need 12.3+)")
        print("  3. Verify you're on Apple Silicon (M1/M2/M3)")
        print("  4. Training will use CPU (slower but still works)")
    
    print("\n" + "="*70)


def main():
    """Run all checks"""
    print("\n" + "#"*70)
    print("PYTORCH MPS CONFIGURATION CHECKER FOR MAC")
    print("#"*70)
    
    # Check availability
    mps_ok = check_mps_availability()
    
    # Test operations
    if mps_ok:
        test_mps_operations()
        benchmark_devices()
    
    # Print recommendations
    print_recommendations()
    
    print("\n" + "#"*70)
    print("CONFIGURATION CHECK COMPLETE")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
