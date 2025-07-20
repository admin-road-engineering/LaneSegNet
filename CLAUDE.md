# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LaneSegNet is a deep learning project for aerial lane detection and segmentation. It combines a FastAPI web service with MMSegmentation models for processing aerial imagery and detecting lane markings. The project supports both traditional U-Net models and advanced Swin Transformer-based segmentation models.

## Key Architecture Components

### FastAPI Web Service (`app/`)
- **`main.py`**: Core FastAPI application with `/detect_lanes` endpoint and health checks
- **`inference.py`**: Inference pipeline using MMSegmentation models
- **`model_loader.py`**: Model initialization and loading logic
- **`schemas.py`**: Pydantic data models for API request/response validation

### Model Configurations (`configs/`)
- **`lanesegnet_r50_8x1_24e_olv2_subset_A.py`**: Main LaneSegNet model configuration with ResNet50 backbone
- **`mmseg/swin-base-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py`**: Swin Transformer configuration for MMSegmentation
- **`ael/upernet_swin-t_512x512_80k_ael.py`**: AEL dataset specific configuration

### Custom MMSegmentation Components (`mmseg_custom/`)
- **`datasets/ael_dataset.py`**: Custom dataset implementation for AEL (Aerial Lane) dataset
- **`datasets/__init__.py`**: Dataset registry

### Data Processing
- **`data/`**: Contains training images, GeoJSON files, and dataset splits
- **`create_ael_masks.py`**: Script for generating segmentation masks from annotations
- **`verify_json_data.py`**: Data validation utilities

## Development Commands

### Docker Development
```bash
# Build the Docker image
docker build -t lanesegnet .

# Run the container
docker run -p 8010:8010 --gpus all lanesegnet

# Create weights directory
./create_weights_dir.sh
```

### Python Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install MMSegmentation and MMCV (as specified in Dockerfile)
pip install --no-binary mmcv "mmcv==2.1.0"
pip install "mmsegmentation==1.2.2"
```

### Running the API
```bash
# Development server
python -m uvicorn app.main:app --reload --port 8010

# Production server (as in Dockerfile)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8010
```

### Testing and Validation
```bash
# Test CUDA and MMCV installation
python test_cuda_mmcv.py

# Verify JSON data integrity
python verify_json_data.py

# Check package versions
python check_versions.py
```

## Model Architecture Details

### LaneSegNet Model
- Uses BEVFormer-style architecture for bird's-eye-view lane detection
- ResNet50 backbone with FPN neck
- Transformer-based decoder with lane attention mechanisms
- Supports temporal reasoning and spatial cross-attention

### MMSegmentation Integration
- Supports Swin Transformer backbones with UperNet heads
- ADE20K pre-trained weights for segmentation
- Custom inference pipeline for aerial imagery processing

### Post-processing Pipeline
1. Extract binary masks for road class (index 6 for ADE20K, index 3 for custom datasets)
2. Resize to original image dimensions
3. Optional skeletonization for centerline extraction
4. Contour detection and polyline approximation
5. Format results as lane segments with coordinate points

## Key Configuration Notes

- **CUDA Support**: Requires NVIDIA GPU with CUDA 12.1+ for optimal performance
- **Image Input**: Accepts RGB images, automatically handles BGR conversion for OpenCV operations
- **Class Indices**: Road class index varies by model (6 for ADE20K, 3 for custom datasets)
- **Memory Requirements**: Large models require significant GPU memory for inference

## File Structure Patterns

- Model weights go in `weights/` directory
- Configuration files follow MMSegmentation conventions
- Custom dataset implementations extend MMSegmentation base classes
- All inference code uses RGB format internally, converting to BGR only for visualization

## Environment Variables

- `FORCE_CUDA=1`: Forces CUDA compilation for MMCV
- `MMCV_WITH_OPS=1`: Enables MMCV operators
- `TORCH_CUDA_ARCH_LIST="8.6"`: Specifies CUDA architecture for compilation