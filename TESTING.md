# LaneSegNet Testing Framework

## Overview

This document describes the comprehensive testing infrastructure implemented in Phase 3.1 for the LaneSegNet aerial lane marking detection system. The testing framework ensures production-ready code quality, performance validation, and prevents critical debug code from reaching deployment.

## Quick Start

### Running Tests

```bash
# Install testing dependencies
pip install -r requirements.txt

# Run all tests with coverage
python scripts/run_tests.py all

# Run specific test types
python scripts/run_tests.py unit          # Unit tests (95%+ coverage)
python scripts/run_tests.py integration   # Integration tests  
python scripts/run_tests.py api          # API endpoint tests
python scripts/run_tests.py load         # Load testing
python scripts/run_tests.py security     # Security scans
python scripts/run_tests.py quality      # Code quality
python scripts/run_tests.py debug-check  # Debug bypass detection
```

### Windows Users
```cmd
scripts\run_tests.bat all
```

## Test Architecture

### Test Organization
```
tests/
├── conftest.py                    # Test fixtures and utilities
├── test_api_endpoints.py          # API endpoint testing
├── test_imagery_acquisition.py    # Imagery provider testing
├── test_coordinate_transform.py   # Geographic coordinate testing
├── test_enhanced_post_processing.py # Post-processing pipeline
└── test_load_testing.py          # Load testing with Locust
```

### Test Categories

#### 1. Unit Tests (95%+ Coverage Target)
- **API Endpoints**: FastAPI endpoint validation
- **Core Modules**: Individual component testing
- **Error Handling**: Exception and edge case validation
- **Mocking**: Isolated testing with dependency mocks

#### 2. Integration Tests
- **End-to-End Workflows**: Complete analysis pipelines
- **Service Dependencies**: External service integration
- **Geographic Accuracy**: Real-world coordinate validation
- **Performance Integration**: Response time validation

#### 3. API Tests
- **Live Server Testing**: Actual API server validation
- **Request/Response Validation**: Schema compliance
- **Error Response Testing**: Proper error handling
- **Health Check Validation**: Service availability

#### 4. Load Tests
- **Concurrent Users**: Multiple simultaneous requests
- **Performance Under Load**: Response time degradation
- **Resource Utilization**: Memory and CPU usage
- **Throughput Testing**: Requests per second validation

#### 5. Security Tests
- **Vulnerability Scanning**: Known security issues
- **Code Security**: Static analysis with Bandit
- **Dependency Security**: Package vulnerability detection
- **Input Validation**: Injection attack prevention

## Test Configuration

### pytest Configuration (pytest.ini)
```ini
[tool:pytest]
testpaths = tests
addopts = 
    --verbose
    --cov=app
    --cov-report=html:htmlcov
    --cov-report=term-missing
    --cov-fail-under=95
markers =
    unit: Unit tests
    integration: Integration tests
    api: API endpoint tests
    slow: Tests that take a long time to run
    requires_gpu: Tests that require GPU/CUDA
    requires_model: Tests that require model weights
```

### Coverage Configuration (pyproject.toml)
```toml
[tool.coverage.run]
source = ["app"]
omit = [
    "*/tests/*",
    "*/test_*",
    "*/__pycache__/*",
]
branch = true

[tool.coverage.report]
show_missing = true
precision = 2
exclude_lines = [
    "pragma: no cover",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
```

## Critical Production Safety Features

### 1. Debug Bypass Detection

**Purpose**: Prevents critical debug code from reaching production

**Implementation**: 
- Automated detection in CI/CD pipeline
- Specific check for `enhanced_post_processing.py:71-73` debug bypass
- Fails deployment if debug patterns found

**Critical Pattern Detected**:
```python
# CRITICAL: This debug bypass MUST be removed before production
debug_markings = lane_markings[:10] if len(lane_markings) > 0 else []
return debug_markings  # Bypasses physics-informed filtering
```

### 2. Performance Validation

**Requirements**:
- Single analysis: <2000ms
- Concurrent load: <1000ms
- Health check: <100ms
- Load testing: 10+ RPS minimum

**Implementation**:
```python
# Performance assertion in tests
assert data["processing_time_ms"] < 1000
```

### 3. Security Integration

**Tools**:
- **Safety**: Known vulnerability detection
- **Bandit**: Security static analysis
- **Semgrep**: Advanced security patterns

**Automation**: Integrated in GitHub Actions pipeline

## Test Fixtures and Utilities

### Common Fixtures (conftest.py)

```python
@pytest.fixture
def test_client():
    """FastAPI test client."""
    with TestClient(app) as client:
        yield client

@pytest.fixture 
def test_coordinates():
    """Standard test coordinates (Brisbane CBD)."""
    return GeographicBounds(
        north=-27.4698, south=-27.4705,
        east=153.0258, west=153.0251
    )

@pytest.fixture
def mock_model():
    """Mock segmentation model."""
    mock = Mock()
    mock.return_value = np.random.randint(0, 12, (512, 512), dtype=np.uint8)
    return mock
```

### Validation Utilities

```python
def assert_valid_coordinates(coords: GeographicBounds):
    """Assert geographic coordinates are valid."""
    assert coords.north > coords.south
    assert coords.east > coords.west
    assert -90 <= coords.south <= 90
    assert -180 <= coords.west <= 180

def assert_valid_infrastructure_element(element: InfrastructureElement):
    """Assert infrastructure element is valid."""
    assert element.class_name is not None
    assert 0 <= element.confidence <= 1
    assert len(element.points) >= 2
```

## CI/CD Pipeline Integration

### GitHub Actions Workflows

#### Main CI Pipeline (`.github/workflows/ci.yml`)
```yaml
jobs:
  code-quality:     # Black, isort, flake8, mypy
  unit-tests:       # Python 3.9-3.11 matrix
  integration-tests: # Service dependencies
  api-tests:        # Live server testing
  security-scan:    # Vulnerability detection
  load-testing:     # Performance validation
  docker-build:     # Container validation
  deployment-check: # Production readiness
```

#### Debug Bypass Detection (`.github/workflows/debug-bypass-check.yml`)
```yaml
jobs:
  detect-debug-code:
    - Check enhanced_post_processing.py for debug bypass
    - Validate physics constraints are active
    - Ensure production safety markers
    - Generate debug bypass report
```

### Deployment Gates

**All gates must pass for deployment**:
- ✅ Code quality checks pass
- ✅ 95%+ test coverage achieved
- ✅ No debug bypass patterns detected
- ✅ Security scans pass
- ✅ Performance requirements met
- ✅ Docker build successful

## Load Testing with Locust

### Test Scenarios

```python
class LaneSegNetUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(5)  # Higher weight = more frequent
    def test_health_check(self):
        """Most frequent - health checks"""
        
    @task(3)
    def test_analyze_road_infrastructure(self):
        """Core functionality testing"""
        
    @task(1)
    def test_visualization_endpoint(self):
        """Resource-intensive testing"""
```

### Performance Benchmarks

```python
PERFORMANCE_REQUIREMENTS = {
    "health_check_max_ms": 100,
    "analysis_max_ms": 2000,
    "concurrent_analysis_max_ms": 1000,
    "visualization_max_ms": 5000,
    "max_error_rate_percent": 5,
    "min_throughput_rps": 10,
}
```

### Running Load Tests

```bash
# Light load (CI testing)
locust -f tests/test_load_testing.py --host=http://localhost:8010 \
       --users 5 --spawn-rate 1 --run-time 60s --headless

# Production load testing
locust -f tests/test_load_testing.py --host=http://localhost:8010 \
       --users 50 --spawn-rate 5 --run-time 300s --headless
```

## Development Workflow

### Pre-commit Checks

```bash
# Before committing code
python scripts/run_tests.py quality --fix  # Auto-fix formatting
python scripts/run_tests.py unit          # Validate functionality
python scripts/run_tests.py debug-check   # Ensure no debug code
```

### Integration Testing

```bash
# Start API server
python -m uvicorn app.main:app --reload --port 8010

# In another terminal
python scripts/run_tests.py api
python scripts/run_tests.py integration
```

### Performance Testing

```bash
# Start server with production settings
python -m uvicorn app.main:app --host 0.0.0.0 --port 8010

# Run load tests
python scripts/run_tests.py load --users 20 --run-time 120s
```

## Troubleshooting

### Common Issues

#### Test Failures

```bash
# Check specific test output
pytest tests/test_api_endpoints.py::TestAnalyzeRoadInfrastructureEndpoint::test_valid_request -v

# Run with debug output
pytest tests/ --tb=long --capture=no
```

#### Coverage Issues

```bash
# Generate detailed coverage report
python scripts/run_tests.py unit --coverage-fail 90

# View HTML coverage report
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html # Windows
```

#### Load Testing Issues

```bash
# Check API server is running
curl http://localhost:8010/health

# Run single-user load test
locust -f tests/test_load_testing.py --host=http://localhost:8010 \
       --users 1 --spawn-rate 1 --run-time 30s --headless
```

### Environment Variables

```bash
# Testing mode (disables GPU, model loading)
export TESTING_MODE=true
export CUDA_VISIBLE_DEVICES=""

# API testing
export API_BASE_URL="http://localhost:8010"

# Load testing configuration
export LOAD_TEST_USERS=10
export LOAD_TEST_DURATION=60s
```

## Quality Metrics

### Current Achievement (Phase 3.1)

- **✅ Test Coverage**: 95%+ target with monitoring
- **✅ API Test Coverage**: 15+ endpoint test cases
- **✅ Performance Validation**: <1000ms under concurrent load
- **✅ Security Integration**: Automated vulnerability scanning
- **✅ Debug Prevention**: Automated debug bypass detection
- **✅ CI/CD Integration**: Complete GitHub Actions pipeline

### Success Criteria for Phase 3.2

- **Maintain 95%+ coverage** during model training integration
- **Performance under load** with new model weights
- **No debug bypasses** in model training code
- **Security validation** of training data handling
- **Integration testing** with 39,094 sample dataset

## Contributing

### Adding New Tests

1. **Create test file**: Follow naming convention `test_*.py`
2. **Add markers**: Use appropriate pytest markers
3. **Mock dependencies**: Use fixtures from `conftest.py`
4. **Document test cases**: Include docstrings and comments
5. **Validate coverage**: Ensure new code is covered

### Test Best Practices

- **Isolated tests**: Each test should be independent
- **Clear assertions**: Use descriptive assertion messages
- **Mock external deps**: Don't rely on external services
- **Performance aware**: Set appropriate timeouts
- **Error testing**: Include failure scenario testing

## Future Enhancements

### Planned for Phase 3.2+

- **Model training tests**: Validation of training pipeline
- **Data quality tests**: Training data validation
- **Performance regression**: Benchmark tracking
- **Visual regression**: Image output validation
- **Cross-platform**: Windows/Linux/macOS validation

This testing framework provides the production-ready foundation needed for confident deployment and continued development of the LaneSegNet system.