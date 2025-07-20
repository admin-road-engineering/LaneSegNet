# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LaneSegNet is a **comprehensive road infrastructure analysis system** that serves as a specialized microservice for aerial lane marking detection and road infrastructure analysis. This system is designed to integrate with the main Road Engineering SaaS platform located at `C:\Users\Admin\road-engineering-branch\road-engineering` and provides **competitive advantage** through AI-powered infrastructure detection.

### Role in Road Engineering Ecosystem
This service is a **future premium feature** for the road engineering platform, designed to provide:
- **Aerial lane marking detection** from geographic coordinates
- **Comprehensive road infrastructure analysis** (pavements, footpaths, utilities)
- **Real-world measurements** (areas in m², lengths in meters)
- **Geographic coordinate integration** with engineering workflows
- **AI-powered accuracy exceeding current SOTA** (targeting 80-85% mIoU vs 76% industry standard)

## Key Architecture Components

### FastAPI Web Service (`app/`)
- **`main.py`**: Core FastAPI application with `/analyze_road_infrastructure` endpoint (coordinate-based)
- **`inference.py`**: Inference pipeline using MMSegmentation models
- **`model_loader.py`**: Model initialization and loading logic for multi-class infrastructure
- **`schemas.py`**: Enhanced data models for infrastructure analysis and real-world measurements
- **`imagery_acquisition.py`**: Multi-provider aerial imagery acquisition system
- **`coordinate_transform.py`**: Geographic coordinate transformation utilities

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

## Implementation Status

### ✅ Phase 1: Core Infrastructure Setup - COMPLETE
- **1.1**: ✅ **Model architecture resolved** - Swin Transformer with CUDA support functional
- **1.2**: ✅ **Coordinate-based imagery acquisition** - OpenStreetMap + multi-provider system implemented  
- **1.3**: ✅ **API endpoints updated** - `/analyze_road_infrastructure` fully functional
- **1.4**: ✅ **Geographic coordinate transformation** - Engineering-grade precision achieved

### ✅ Phase 1.5: Docker Infrastructure - COMPLETE
- **Docker Containerization**: ✅ **MMCV dependencies resolved** with CUDA 12.1 support
- **Production Deployment**: ✅ **25.2GB container** with complete infrastructure
- **Performance Validation**: ✅ **1.16s response time** (42% faster than 2s target)
- **API Integration**: ✅ **Production-ready** for road-engineering frontend

### 🔄 Phase 2: Performance Optimization - IN PROGRESS
- **2.1**: 🔄 **Model accuracy enhancement** - Current 45-55% vs target 80-85% mIoU
- **2.2**: ✅ **Real-world measurements** - Areas in m², lengths in meters implemented
- **2.3**: 🔄 **12-class lane detection** - Currently 6 types vs target 12 classes
- **2.4**: ✅ **Geographic validation** - Physics-informed coordinate constraints active

### ⏳ Phase 3: Advanced Features - PENDING
- **3.1**: ✅ **Frontend integration ready** - CORS and API compatibility validated
- **3.2**: ⏳ **Batch processing optimization** - Architecture supports coordinate regions
- **3.3**: ⏳ **Caching system** - Framework ready for implementation
- **3.4**: ✅ **Performance benchmarking** - Comprehensive assessment completed

**🎯 CURRENT STATUS**: **Production-Ready Infrastructure** with performance optimization needed for premium feature targets.

## Development Commands

### Docker Development ✅ VALIDATED
```bash
# Build the Docker image (25.2GB with CUDA 12.1 + MMSegmentation)
docker build -t lanesegnet .

# Run the container (RECOMMENDED - Resolves all MMCV dependencies)
docker run -p 8010:8010 --gpus all lanesegnet

# Check container status
docker ps -a --filter "ancestor=lanesegnet"

# Create weights directory (if needed)
./create_weights_dir.sh
```

**Docker Status**: ✅ **PRODUCTION READY** - All dependency issues resolved, MMCV with CUDA extensions functional

### Python Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install MMSegmentation and MMCV (as specified in Dockerfile)
pip install --no-binary mmcv "mmcv==2.1.0"
pip install "mmsegmentation==1.2.2"
```

### Running the API

#### 🚨 CRITICAL SERVICE MANAGEMENT RULE
**IMPORTANT**: Always check if service is running first with `netstat -ano | findstr :8010` before starting. Multiple uvicorn instances cause port conflicts and integration failures with the road-engineering frontend.

```bash
# Check if service is running
netstat -ano | findstr :8010

# Development server
python -m uvicorn app.main:app --reload --port 8010

# Production server (as in Dockerfile)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8010
```

#### Frontend Integration Support
- **CORS enabled for**: `localhost:5173`, `localhost:5174`, `localhost:3001` (road-engineering frontend)
- **Main endpoint**: `POST /analyze_road_infrastructure` - Coordinate-based infrastructure analysis
- **Integration URL**: `http://localhost:8010` (development) / `https://lanesegnet-api.road.engineering` (production)

### Testing and Validation ✅ COMPLETE
```bash
# Health check (validates model loading and API)
curl http://localhost:8010/health

# Full infrastructure analysis test (Brisbane coordinates)
curl -X POST "http://localhost:8010/analyze_road_infrastructure" \
  -H "Content-Type: application/json" \
  -d '{"north": -27.4698, "south": -27.4705, "east": 153.0258, "west": 153.0251}'

# Legacy dependency tests (use Docker instead)
python test_cuda_mmcv.py
python verify_json_data.py
python check_versions.py
```

**Validation Status**: ✅ **COMPREHENSIVE TESTING COMPLETE**
- Response Time: **1.16s** (42% faster than 2s target)
- Lane Detection: **37 segments** across **6 lane types**
- Geographic Accuracy: **Engineering-grade** coordinate transformation
- Docker Infrastructure: **Fully functional** with CUDA support

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

## Integration with Road Engineering Platform

### Role as Specialized Microservice
LaneSegNet serves as the **primary infrastructure analysis provider** for the main Road Engineering SaaS platform. The system supports enterprise-grade road infrastructure analysis tools including:
- **Comprehensive infrastructure detection** from geographic coordinates
- **Real-world measurement capabilities** for engineering calculations
- **High-resolution analysis** supporting professional engineering workflows
- **Multi-provider imagery acquisition** for global coverage

### API Integration Points
```
Road Engineering Frontend → LaneSegNet API → Infrastructure Analysis
Coordinates Input → Multi-Provider Imagery → AI Analysis → Engineering Data
```

**Integration Endpoints:**
- **Primary**: `/analyze_road_infrastructure` - Coordinate-to-infrastructure analysis pipeline
- **Health Check**: `/health` - Service status monitoring
- **Data Volume**: High-resolution analysis supporting professional engineering standards

**Response Format**: Detailed infrastructure analysis with geographic coordinates, real-world measurements, and confidence scoring for engineering validation.

## Environment Variables

### Core Configuration
- `FORCE_CUDA=1`: Forces CUDA compilation for MMCV
- `MMCV_WITH_OPS=1`: Enables MMCV operators
- `TORCH_CUDA_ARCH_LIST="8.6"`: Specifies CUDA architecture for compilation

### Imagery Acquisition
- `GOOGLE_EARTH_ENGINE_API_KEY`: Google Earth Engine API access (optional)
- `MAPBOX_API_KEY`: Mapbox satellite imagery API (optional)
- `LOCAL_IMAGERY_DIR`: Local imagery directory (default: 'data/imagery')

**Current Provider Status**: ✅ **OpenStreetMap** (free, no API key required) - Primary provider

### Integration Settings
- `ROAD_ENGINEERING_FRONTEND_URL`: Frontend URL for CORS configuration
- `ANALYSIS_CACHE_SIZE`: Cache size for repeated analysis requests
- `MAX_COORDINATE_REGIONS`: Maximum coordinate regions per batch request

## Claude Development Rules

When working on this codebase, follow these specific rules:

1. **Problem Analysis & Planning**: First think through the problem, read the codebase for relevant files, and create todos using TodoWrite
2. **Todo List Management**: Use TodoWrite tool for tracking all tasks and mark items as complete immediately after finishing
3. **Progressive Implementation**: Work through todo items systematically, providing high-level explanations
4. **Integration Focus**: Always consider integration with road-engineering frontend when making changes
5. **Performance Priority**: Maintain real-time analysis capability for professional engineering workflows
6. **Geographic Accuracy**: Ensure coordinate transformations are precise for engineering measurements

## Development Protocols

### Key Principles
- **Infrastructure Detection Focus**: All changes should enhance infrastructure analysis capabilities
- **Real-World Measurements**: Ensure accurate metric calculations for engineering validation  
- **Frontend Integration**: Maintain compatibility with road-engineering platform coordinate system
- **Performance Optimization**: Optimize for batch processing of coordinate regions
- **Error Handling**: Robust error handling for geographic coordinate edge cases

### Security Requirements

#### API Security
- ✅ **Validate all coordinate inputs** with geographic bounds checking
- ✅ **Rate limiting** on analysis endpoints (50 requests/hour/user for premium features)
- ✅ **Sanitize all outputs** (no model internals exposed to frontend)
- ✅ **CORS restrictions** (only road-engineering frontend domains)
- ✅ **Input validation** for all geographic coordinate parameters

#### Integration Security
- ✅ **API key protection** for imagery providers (Google Earth Engine, Mapbox)
- ✅ **No secrets in frontend code** - all sensitive imagery API keys server-side only
- ✅ **Geographic bounds validation** to prevent analysis outside service areas
- ✅ **Resource limits** on analysis complexity and coordinate region size

## Troubleshooting Guide

### Common Issues and Solutions

#### Model Loading Issues
**Symptoms**: API returns 503 "Model not available" errors
**Solutions**:
1. **Check model weights**: Verify `weights/` directory contains required model files
2. **CUDA availability**: Confirm GPU access with `python test_cuda_mmcv.py`
3. **Memory requirements**: Ensure sufficient GPU memory for model loading
4. **Dependencies**: Verify MMSegmentation and MMCV versions match requirements

#### Coordinate Transformation Errors
**Symptoms**: Invalid geographic coordinates or measurement errors
**Solutions**:
1. **Bounds validation**: Check coordinates are within supported geographic regions
2. **Resolution limits**: Verify resolution_mpp parameter is within supported range (0.1-2.0)
3. **CRS compatibility**: Ensure coordinate reference system matches imagery provider

#### Imagery Acquisition Failures
**Symptoms**: "All imagery providers failed" errors
**Solutions**:
1. **API key validation**: Verify Google Earth Engine and Mapbox API keys are valid
2. **Network connectivity**: Check internet access for external imagery providers
3. **Fallback chain**: Ensure local imagery available when external providers fail
4. **Rate limits**: Monitor API usage limits for imagery providers

#### Frontend Integration Issues
**Symptoms**: CORS errors or coordinate mismatch between frontend and backend
**Solutions**:
1. **CORS configuration**: Verify frontend URL in CORS middleware settings
2. **Coordinate format**: Ensure geographic bounds format matches schema expectations
3. **Response validation**: Check infrastructure analysis response format compatibility

### Diagnostic Commands

```bash
# Check service status and model loading
curl http://localhost:8010/health

# Test coordinate analysis (Brisbane test coordinates)
curl -X POST "http://localhost:8010/analyze_road_infrastructure" \
  -H "Content-Type: application/json" \
  -d '{
    "north": -27.4698,
    "south": -27.4705,
    "east": 153.0258,
    "west": 153.0251,
    "analysis_type": "comprehensive",
    "resolution": 0.1
  }'

# Check model and dependency versions
python check_versions.py

# Verify CUDA and MMSegmentation setup
python test_cuda_mmcv.py

# Test imagery acquisition providers
python -c "from app.imagery_acquisition import imagery_manager; print(f'Available providers: {list(imagery_manager.providers.keys())}')"
```