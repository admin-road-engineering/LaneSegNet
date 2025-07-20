# LaneSegNet Production Deployment Guide

## API Provider Cost Analysis

### Mapbox Satellite API (Primary Provider)

**Pricing Structure (2024):**
- **Base Plan**: $50/month minimum
- **Request-based billing**: Tile-based pricing model
- **Static Images API**: Used for coordinate-based imagery acquisition

**Cost Calculation for LaneSegNet:**
```
Coordinate Analysis Request → Mapbox Static API Call
├── URL Format: /styles/v1/mapbox/satellite-v9/static/[bbox]/WxH@2x
├── Cost per request: ~$0.0025 - $0.005 per tile
└── Estimated monthly cost: 1000 requests = $2.50 - $5.00
```

**Production Configuration:**
```python
# Environment variables required
MAPBOX_API_KEY=pk.your_production_key_here

# Cost optimization settings
MAPBOX_MAX_DIMENSION=1280  # Per API limits
MAPBOX_RETRY_ATTEMPTS=3
MAPBOX_TIMEOUT_SECONDS=30
```

**Cost Management Strategy:**
1. **Caching**: Implement coordinate-based caching to reduce repeated requests
2. **Size optimization**: Use optimal image dimensions for analysis needs
3. **Fallback chain**: Local → Mapbox → GEE to minimize external costs
4. **Rate limiting**: Prevent cost overruns from excessive usage

### Google Earth Engine (Secondary Provider)

**Enterprise Pricing Model:**
- **Individual/SMB tier**: $X per EECU (Compute units)
- **Enterprise tier**: Custom pricing based on usage
- **Commercial requirements**: Google Cloud project required

**Implementation Status**: ⚠️ Placeholder implementation
- **Immediate need**: Medium priority (Mapbox covers primary use cases)
- **Implementation effort**: 2-3 days for full integration
- **Cost benefit**: Higher resolution for premium customers

### Production Cost Estimates

**Monthly Operating Costs (Conservative):**
```
Service Component               Cost Range
────────────────────────────────────────
Mapbox API (1000 requests)     $2.50 - $5.00
Google Earth Engine            $10 - $50 (if enabled)
Local development fallback     $0
────────────────────────────────────────
Total External API Costs      $2.50 - $55.00
```

**Scale Projections:**
- **10,000 requests/month**: $25 - $100
- **100,000 requests/month**: $250 - $1,000
- **1M requests/month**: Enterprise negotiation required

## Performance Specifications

### Response Time Targets

**Current Implementation:**
- **Target**: <2 seconds for 1km² coordinate analysis
- **Components breakdown**:
  ```
  Imagery acquisition:    200-800ms (depending on provider)
  Model inference:        300-600ms (MMSegmentation + CV)
  Coordinate transform:   50-100ms
  Response formatting:    50-100ms
  ────────────────────────────────────
  Total estimated:        600-1600ms
  ```

### Scalability Architecture

**Concurrent Request Handling:**
- **FastAPI async**: Supports concurrent imagery acquisition
- **Model loading**: Single model instance (GPU memory optimization)
- **Provider fallback**: Parallel provider attempts for reliability

**Resource Requirements:**
```
Component               Requirement
─────────────────────────────────
GPU Memory             4-8GB (MMSegmentation model)
RAM                    8-16GB (image processing)
Storage                10GB (model weights + local imagery)
Network                Stable connection for API providers
```

## Security & API Key Management

### Production Environment Variables

**Required for Production:**
```bash
# Primary imagery provider
MAPBOX_API_KEY=pk.your_production_key_here

# Optional secondary provider
GOOGLE_EARTH_ENGINE_API_KEY=your_gee_key_here

# Service configuration
LANESEGNET_ENVIRONMENT=production
CORS_ORIGINS=https://road.engineering,https://app.road.engineering
MAX_COORDINATE_AREA_KM2=100
ENABLE_API_COST_MONITORING=true
```

**Security Best Practices:**
1. **API Key Rotation**: Monthly rotation of production keys
2. **Rate Limiting**: 50 requests/hour per user for cost control
3. **CORS Restrictions**: Only allow road-engineering frontend domains
4. **Input Validation**: Strict coordinate bounds checking
5. **Cost Monitoring**: Alert when API costs exceed thresholds

### Deployment Checklist

**Pre-Production:**
- [ ] Model weights deployed to production environment
- [ ] Mapbox API key configured and tested
- [ ] CORS origins updated for production domains
- [ ] Health check endpoint responding correctly
- [ ] Performance benchmarks meet <2 second target
- [ ] Cost monitoring alerts configured

**Production Readiness:**
- [ ] Integration testing with road-engineering frontend complete
- [ ] Load testing with concurrent requests verified
- [ ] Error handling tested with provider failures
- [ ] Logging and monitoring configured
- [ ] Backup and recovery procedures documented

## Monitoring & Alerting

**Key Metrics to Monitor:**
1. **Response times**: 95th percentile <2 seconds
2. **API costs**: Daily/monthly spend tracking
3. **Error rates**: <1% imagery acquisition failures
4. **Model performance**: Detection confidence scores
5. **Concurrent usage**: Peak concurrent request handling

**Alert Thresholds:**
```
Metric                  Alert Threshold
────────────────────────────────────
Response time           >3 seconds (95th percentile)
Daily API cost          >$10
Error rate              >5% over 1 hour
GPU memory usage        >90%
Concurrent requests     >50 simultaneous
```

## Integration with Road Engineering Platform

**API Endpoints:**
- **Primary**: `POST /analyze_road_infrastructure`
- **Health**: `GET /health`
- **Service URL**: `https://lanesegnet-api.road.engineering`

**Frontend Integration:**
```javascript
// Example integration from road-engineering frontend
const response = await fetch('https://lanesegnet-api.road.engineering/analyze_road_infrastructure', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    north: -27.4698,
    south: -27.4705, 
    east: 153.0258,
    west: 153.0251,
    analysis_type: 'comprehensive',
    resolution: 0.1
  })
});
```

## Troubleshooting Guide

**Common Production Issues:**

1. **High API Costs**
   - Check caching implementation
   - Verify coordinate deduplication
   - Monitor request patterns for abuse

2. **Slow Response Times**
   - Check imagery provider performance
   - Verify GPU availability
   - Monitor concurrent request load

3. **Provider Failures**
   - Verify API key validity
   - Check provider service status
   - Confirm fallback chain operation

**Emergency Procedures:**
- **Cost overrun**: Disable external providers, use local fallback
- **Service failure**: Health check monitoring with automatic restart
- **Performance degradation**: Reduce concurrent request limits

This deployment guide ensures production-ready operation with cost control, performance monitoring, and proper integration with the Road Engineering SaaS platform.