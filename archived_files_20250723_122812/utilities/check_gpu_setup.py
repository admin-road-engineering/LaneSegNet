#!/usr/bin/env python3
"""
GPU Setup Diagnostic Script
Check CUDA availability and GPU configuration
"""

import torch
import sys
from pathlib import Path

def check_gpu_setup():
    """Comprehensive GPU setup check."""
    print("=== GPU Setup Diagnostic ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    # Basic CUDA check
    print("1. CUDA Availability:")
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   cuDNN version: {torch.backends.cudnn.version()}")
        print(f"   Device count: {torch.cuda.device_count()}")
        print()
        
        # Device details
        print("2. GPU Device Information:")
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   Device {i}: {device_name}")
            print(f"   Total memory: {memory_total:.1f}GB")
            
            # Memory usage
            if i == 0:  # Check primary GPU
                torch.cuda.set_device(i)
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"   Allocated: {memory_allocated:.2f}GB")
                print(f"   Reserved: {memory_reserved:.2f}GB")
                print(f"   Free: {memory_total - memory_reserved:.2f}GB")
        print()
        
        # Test tensor operations
        print("3. GPU Tensor Test:")
        try:
            # Simple tensor operation
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("   Basic tensor operations: PASS")
            
            # Test neural network operations
            import torch.nn as nn
            model = nn.Linear(1000, 100).cuda()
            output = model(x)
            print("   Neural network operations: PASS")
            
            # Cleanup
            del x, y, z, model, output
            torch.cuda.empty_cache()
            print("   Memory cleanup: PASS")
            
        except Exception as e:
            print(f"   GPU operations failed: {e}")
        
        print()
    else:
        print("   No CUDA devices found")
        print()
        
        # Diagnose why CUDA might not be available
        print("2. CUDA Troubleshooting:")
        
        # Check if CUDA is installed
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("   nvidia-smi available: YES")
                print("   NVIDIA drivers: Installed")
                # Parse GPU info from nvidia-smi
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'NVIDIA' in line and 'Driver' in line:
                        print(f"   {line.strip()}")
            else:
                print("   nvidia-smi available: NO")
                print("   NVIDIA drivers: Missing or not in PATH")
        except FileNotFoundError:
            print("   nvidia-smi: Not found")
            print("   NVIDIA drivers: Not installed or not in PATH")
        
        # Check PyTorch CUDA installation
        if hasattr(torch.version, 'cuda') and torch.version.cuda:
            print(f"   PyTorch CUDA version: {torch.version.cuda}")
            print("   PyTorch: Compiled with CUDA support")
        else:
            print("   PyTorch: CPU-only version installed")
            print("   Recommendation: Install PyTorch with CUDA support")
    
    # Check if our training model would fit
    print("4. Model Memory Requirements:")
    try:
        # Estimate memory for our model
        if cuda_available:
            # Approximate memory needed for training
            model_params = 1566644  # Our model parameters
            batch_size = 8
            input_size = 512 * 512 * 3  # Image dimensions
            
            # Rough estimates (in GB)
            model_memory = (model_params * 4) / 1024**3  # 4 bytes per float32
            batch_memory = (batch_size * input_size * 4) / 1024**3
            gradient_memory = model_memory  # Gradients same size as model
            optimizer_memory = model_memory * 2  # Adam stores momentum
            
            total_estimated = model_memory + batch_memory + gradient_memory + optimizer_memory
            
            print(f"   Estimated model memory: {model_memory:.3f}GB")
            print(f"   Estimated batch memory: {batch_memory:.3f}GB")
            print(f"   Estimated total training: {total_estimated:.3f}GB")
            
            available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   Available GPU memory: {available_memory:.1f}GB")
            
            if total_estimated < available_memory * 0.8:  # 80% threshold
                print("   Memory assessment: SUFFICIENT for GPU training")
            else:
                print("   Memory assessment: MAY BE TIGHT (consider smaller batch size)")
        else:
            print("   Cannot assess - no CUDA available")
    except:
        print("   Memory assessment: Unable to calculate")
    
    print()
    
    # Recommendations
    print("5. Recommendations:")
    if cuda_available:
        print("   GPU is available and ready for training!")
        print("   To use GPU in future training:")
        print("   - Ensure model.to('cuda') and tensors.to('cuda')")
        print("   - Current training is CPU-only by design or error")
    else:
        print("   To enable GPU training:")
        print("   1. Install NVIDIA drivers (if not installed)")
        print("   2. Install CUDA toolkit")
        print("   3. Install PyTorch with CUDA support:")
        print("      pip uninstall torch torchvision torchaudio")
        print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("   4. Restart Python environment")
    
    print()
    return cuda_available

if __name__ == "__main__":
    gpu_available = check_gpu_setup()
    
    if gpu_available:
        print("STATUS: GPU ready for accelerated training!")
    else:
        print("STATUS: CPU-only training (current setup)")