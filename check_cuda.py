"""
CUDA and GPU Availability Check
Checks PyTorch CUDA support, GPU details, and system compatibility
"""

import sys

def check_cuda():
    print("=" * 70)
    print("CUDA & GPU Availability Check")
    print("=" * 70)
    
    # Check PyTorch
    try:
        import torch
        print(f"\n✓ PyTorch installed: {torch.__version__}")
    except ImportError:
        print("\n✗ PyTorch not installed")
        print("  Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        sys.exit(1)
    
    # Check CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print("✓ CUDA is available - GPU training enabled!")
        
        # CUDA version
        print(f"\nCUDA version (PyTorch): {torch.version.cuda}")
        
        # cuDNN
        print(f"cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        
        # GPU count and details
        gpu_count = torch.cuda.device_count()
        print(f"\nNumber of GPUs: {gpu_count}")
        
        for i in range(gpu_count):
            print(f"\n--- GPU {i} ---")
            print(f"Name: {torch.cuda.get_device_name(i)}")
            
            # Memory info
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"Total memory: {total_mem:.2f} GB")
            
            if torch.cuda.is_initialized():
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"Allocated memory: {allocated:.2f} GB")
                print(f"Reserved memory: {reserved:.2f} GB")
            
            # Compute capability
            capability = torch.cuda.get_device_capability(i)
            print(f"Compute capability: {capability[0]}.{capability[1]}")
        
        # Current device
        print(f"\nCurrent CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        
        # Test GPU computation
        print("\n--- Testing GPU Computation ---")
        try:
            x = torch.rand(1000, 1000, device='cuda')
            y = torch.rand(1000, 1000, device='cuda')
            z = torch.matmul(x, y)
            print("✓ GPU computation test passed")
            del x, y, z
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ GPU computation test failed: {e}")
    
    else:
        print("✗ CUDA is NOT available - CPU-only mode")
        print(f"\nPyTorch version: {torch.__version__}")
        
        if "+cpu" in torch.__version__:
            print("\n⚠ You have the CPU-only version of PyTorch installed")
            print("\nTo enable GPU training:")
            print("1. Uninstall current PyTorch:")
            print("   pip uninstall torch torchvision torchaudio -y")
            print("\n2. Install CUDA-enabled PyTorch:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        else:
            print("\n⚠ PyTorch with CUDA support is installed but CUDA is not detected")
            print("Possible issues:")
            print("  - NVIDIA GPU drivers not installed")
            print("  - No NVIDIA GPU in system")
            print("  - GPU is disabled in BIOS")
    
    # Check numpy compatibility
    print("\n" + "=" * 70)
    print("Dependency Check")
    print("=" * 70)
    
    try:
        import numpy as np
        print(f"\n✓ NumPy: {np.__version__}")
        
        # Test torch-numpy compatibility
        try:
            arr = np.array([1, 2, 3])
            tensor = torch.from_numpy(arr)
            print("✓ NumPy-PyTorch compatibility: OK")
        except Exception as e:
            print(f"✗ NumPy-PyTorch compatibility issue: {e}")
            print("  Solution: pip install 'numpy<2'")
    except ImportError:
        print("✗ NumPy not installed")
    
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not installed")
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics (YOLOv8): installed")
    except ImportError:
        print("✗ Ultralytics not installed")
    
    print("\n" + "=" * 70)
    print("Check complete!")
    print("=" * 70)


if __name__ == "__main__":
    check_cuda()
