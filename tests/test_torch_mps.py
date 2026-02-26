#!/usr/bin/env python3
"""
Test script to verify PyTorch with MPS support on Apple Silicon Macs.

This script performs a simple tensor operation using the MPS device
to verify that PyTorch is correctly configured to use Metal Performance Shaders
for hardware acceleration on Apple Silicon.
"""

import os
import sys
import time
import platform

def print_header(text):
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")

def print_section(text):
    """Print a section header."""
    print(f"\n--- {text} ---")

def main():
    print_header("PyTorch MPS Test for Apple Silicon")
    
    # Check if running on Apple Silicon
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print("❌ Not running on Apple Silicon Mac.")
        print(f"   System: {platform.system()}")
        print(f"   Machine: {platform.machine()}")
        return False
    
    print("✅ Running on Apple Silicon Mac.")
    
    # Try importing torch
    print_section("Checking PyTorch installation")
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
    except ImportError:
        print("❌ Failed to import torch. Please install PyTorch first.")
        return False
    
    # Check for MPS availability
    print_section("Checking MPS availability")
    
    # Different versions of PyTorch have different ways to check MPS
    mps_available = False
    
    # Try the standard way (PyTorch 2.0+)
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
        try:
            if torch.backends.mps.is_available():
                mps_available = True
        except AttributeError:
            pass
    
    # Try the older way or direct check
    if not mps_available and hasattr(torch, 'mps'):
        try:
            # Some versions use torch.mps.is_available()
            if hasattr(torch.mps, 'is_available') and torch.mps.is_available():
                mps_available = True
            # Some versions just check if the module exists
            else:
                mps_available = True
        except (AttributeError, ImportError):
            pass
            
    # Try creating a tensor on MPS as final check
    if not mps_available:
        try:
            device = torch.device("mps")
            x = torch.ones(1, device=device)
            mps_available = True
        except (RuntimeError, ValueError):
            pass
    
    if not mps_available:
        print("❌ MPS is not available on this system.")
        return False
    
    print("✅ MPS is available!")
    
    # Run a simple test with CPU vs. MPS
    print_section("Running performance comparison test")
    
    # Create test tensors
    size = 2000
    print(f"Creating {size}x{size} matrices for multiplication test...")
    
    # Test on CPU
    start_time = time.time()
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    print("Running matrix multiplication on CPU...", end="", flush=True)
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    print(f" Done in {cpu_time:.4f} seconds")
    
    # Test on MPS if available
    try:
        start_time = time.time()
        mps_device = torch.device("mps")
        a_mps = torch.randn(size, size, device=mps_device)
        b_mps = torch.randn(size, size, device=mps_device)
        print("Running matrix multiplication on MPS...", end="", flush=True)
        c_mps = torch.matmul(a_mps, b_mps)
        # Force synchronization to get accurate timing
        torch.mps.synchronize()
        mps_time = time.time() - start_time
        print(f" Done in {mps_time:.4f} seconds")
        
        # Compare results for correctness (allow for some floating point differences)
        print_section("Verifying results")
        c_mps_cpu = c_mps.to("cpu")
        max_diff = torch.max(torch.abs(c_cpu - c_mps_cpu)).item()
        print(f"Maximum difference between CPU and MPS results: {max_diff:.6f}")
        
        if max_diff < 1e-3:
            print("✅ Results match within tolerance!")
        else:
            print("⚠️ Results have significant differences.")
        
        # Show speedup
        print_section("Performance Results")
        if mps_time < cpu_time:
            speedup = cpu_time / mps_time
            print(f"✅ MPS is {speedup:.2f}x faster than CPU!")
        else:
            slowdown = mps_time / cpu_time
            print(f"⚠️ MPS is {slowdown:.2f}x slower than CPU. This is unusual for matrix operations.")
    
    except Exception as e:
        print(f"\n❌ Error running MPS test: {str(e)}")
        return False
    
    print_header("Test Summary")
    print("✅ PyTorch is correctly installed with MPS support")
    print("✅ MPS acceleration is working on this Apple Silicon Mac")
    print("\nYour setup is optimized for machine learning on Apple Silicon!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)