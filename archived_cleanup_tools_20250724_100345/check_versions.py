# check_versions.py
import torch
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"CUDA Available: True")
    # This is the CUDA version PyTorch was built with - THIS IS WHAT YOU NEED
    print(f"CUDA Version (PyTorch Built With): {torch.version.cuda}") 
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA Available: False (Running on CPU or CUDA not detected by PyTorch)")

# Optional: Check torchvision version
try:
    import torchvision
    print(f"Torchvision Version: {torchvision.__version__}")
except ImportError:
    print("Torchvision not found.")

try:
    import mmcv
    print(f"MMCV Version: {mmcv.__version__}")
except ImportError:
    print("MMCV not found.")

try:
    import mmseg
    print(f"MMSegmentation Version: {mmseg.__version__}")
except ImportError:
    print("MMSegmentation not found.") 