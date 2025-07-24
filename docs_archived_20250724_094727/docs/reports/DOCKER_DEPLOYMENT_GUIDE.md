# LaneSegNet Docker Deployment Guide

## Prerequisites

### 1. Start Docker Desktop
**REQUIRED**: Start Docker Desktop before proceeding
- Open Docker Desktop application
- Wait for Docker engine to start (Docker icon should be green)
- Verify with: `docker --version`

### 2. GPU Support (NVIDIA)
For full performance with CUDA acceleration:
- Install NVIDIA Docker runtime
- Ensure NVIDIA GPU drivers are installed
- Verify with: `nvidia-smi`

## Quick Start Commands

### Build Container
```bash
# Build the LaneSegNet container (takes 10-15 minutes first time)
docker build -t lanesegnet .
```

### Run Container  
```bash
# Run with GPU support (recommended)
docker run -p 8010:8010 --gpus all lanesegnet

# Run CPU-only (fallback)
docker run -p 8010:8010 lanesegnet
```

### Development Mode
```bash
# Run with local code mounting for development
docker run -p 8010:8010 --gpus all -v "%cd%\app:/app/app" lanesegnet
```

## Validation Test Sequence

Once the container is running, execute these validation tests:

### 1. Health Check
```bash
curl http://localhost:8010/health
# Expected: {"status":"ok","model_loaded":true}
```

### 2. Basic Infrastructure Analysis
```bash
curl -X POST http://localhost:8010/analyze_road_infrastructure \
  -H "Content-Type: application/json" \
  -d "{
    \"north\": -27.4698,
    \"south\": -27.4705,
    \"east\": 153.0258,
    \"west\": 153.0251,
    \"analysis_type\": \"comprehensive\",
    \"resolution\": 0.1
  }"
```

### 3. Performance Benchmarking
```bash
# Test response time (should be <2 seconds)
curl -w "@%{time_total}s\n" -X POST http://localhost:8010/analyze_road_infrastructure \
  -H "Content-Type: application/json" \
  -d "{\"north\": -27.4698, \"south\": -27.4705, \"east\": 153.0258, \"west\": 153.0251}"
```

### 4. Enhanced CV Pipeline Validation
```bash
# Test with visualization enabled
curl -X POST "http://localhost:8010/analyze_road_infrastructure?visualize=true" \
  -H "Content-Type: application/json" \
  -d "{\"north\": -27.4698, \"south\": -27.4705, \"east\": 153.0258, \"west\": 153.0251}" \
  --output lane_detection_result.jpg
```

## Container Architecture

### Dependencies Resolution
The Docker container resolves all MMCV dependency issues:
- ✅ **PyTorch 2.1.2 + CUDA 12.1**: Pre-installed with GPU support
- ✅ **MMCV 2.1.0**: Compiled from source with CUDA extensions
- ✅ **MMSegmentation 1.2.2**: Full segmentation framework
- ✅ **Enhanced CV Pipeline**: OpenCV + scikit-image for post-processing

### Environment Variables
```bash
# CUDA compilation flags (set in Dockerfile)
FORCE_CUDA=1
MMCV_WITH_OPS=1
TORCH_CUDA_ARCH_LIST="8.6"
```

### Volume Mounts
```bash
# Optional: Mount local data for testing
-v "%cd%\data:/app/data"

# Optional: Mount weights for model updates  
-v "%cd%\weights:/app/weights"

# Development: Mount app code
-v "%cd%\app:/app/app"
```

## Expected Performance Metrics

### Response Time Targets
- **Cold start**: 5-10 seconds (model loading)
- **Warm requests**: <2 seconds (target validation)
- **1km² analysis**: <1.5 seconds average

### Detection Capabilities
- **Lane marking types**: 12 classes (white/yellow solid/dashed, road edges, center lines)
- **Computer vision enhancement**: HSV analysis + Hough line detection
- **Geographic accuracy**: Meter-level precision for engineering applications

### Resource Usage
- **GPU Memory**: 4-8GB (MMSegmentation model)
- **RAM**: 8-16GB (image processing)
- **Storage**: 10GB (model weights + dependencies)

## Troubleshooting

### Container Build Issues
```bash
# Check Docker Desktop status
docker info

# Clear Docker cache if build fails
docker system prune -f

# View build logs for debugging
docker build -t lanesegnet . --no-cache
```

### Runtime Issues
```bash
# Check container logs
docker logs <container_id>

# Debug inside container
docker run -it lanesegnet bash

# Test CUDA availability
docker run --gpus all lanesegnet python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### API Issues
```bash
# Test model loading
curl http://localhost:8010/health

# Verify coordinate validation
curl -X POST http://localhost:8010/analyze_road_infrastructure \
  -H "Content-Type: application/json" \
  -d "{\"north\": 0, \"south\": 0, \"east\": 0, \"west\": 0}"
# Should return 400 error for invalid bounds
```

## Integration with Road Engineering Frontend

### CORS Configuration
Container is pre-configured for frontend integration:
```javascript
// Allowed origins in production
const allowedOrigins = [
  'http://localhost:5173',  // Vite dev server
  'http://localhost:5174',  // Alternative port
  'http://localhost:3001',  // React dev server
  'https://road.engineering',
  'https://app.road.engineering'
];
```

### API Endpoint
```javascript
// Frontend integration example
const analyzeInfrastructure = async (bounds) => {
  const response = await fetch('http://localhost:8010/analyze_road_infrastructure', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(bounds)
  });
  return response.json();
};
```

## Production Deployment

### Container Registry
```bash
# Tag for production
docker tag lanesegnet:latest registry.road.engineering/lanesegnet:v1.0.0

# Push to registry
docker push registry.road.engineering/lanesegnet:v1.0.0
```

### Environment Variables for Production
```bash
# Required for production
MAPBOX_API_KEY=pk.your_production_key
GOOGLE_EARTH_ENGINE_API_KEY=your_gee_key
LANESEGNET_ENVIRONMENT=production
MAX_COORDINATE_AREA_KM2=100
ENABLE_API_COST_MONITORING=true
```

### Health Monitoring
```bash
# Container health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:8010/health || exit 1
```

## Expected Validation Results

After Docker deployment, you should see:

### ✅ Successful Model Loading
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "Swin Transformer",
  "classes_supported": 12,
  "cuda_available": true
}
```

### ✅ Infrastructure Analysis Response
```json
{
  "infrastructure_elements": [
    {
      "class": "single_white_solid",
      "confidence": 0.95,
      "geographic_points": [...],
      "length_meters": 45.2,
      "area_sqm": 12.8
    }
  ],
  "processing_time_ms": 1234.5,
  "analysis_summary": {
    "total_elements": 8,
    "total_lane_length_m": 156.7
  }
}
```

This Docker approach resolves all MMCV dependency issues and provides a production-ready LaneSegNet deployment with enhanced computer vision capabilities.