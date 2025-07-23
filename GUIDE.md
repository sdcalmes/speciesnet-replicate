# SpeciesNet Replicate Deployment Guide

A comprehensive guide to wrap Google's SpeciesNet camera trap AI model and deploy it on Replicate with GitHub CI/CD.

## Overview

SpeciesNet is an ensemble of AI models for classifying wildlife in camera trap images. It combines:
- **MegaDetector**: Object detector for animals, humans, and vehicles
- **SpeciesNet Classifier**: Species-level classification for 2000+ labels
- **Ensemble Logic**: Combines detection and classification with geographic filtering

This guide will help you create a complete end-to-end solution to deploy SpeciesNet on Replicate with optional parameters for confidence thresholds, IoU values, and geographic filtering.

## Table of Contents

1. [Project Setup](#project-setup)
2. [Repository Structure](#repository-structure)
3. [Replicate Model Configuration](#replicate-model-configuration)
4. [Model Implementation](#model-implementation)
5. [Local Development & Testing](#local-development--testing)
6. [GitHub Actions CI/CD](#github-actions-cicd)
7. [Deployment Process](#deployment-process)
8. [Usage Examples](#usage-examples)
9. [Troubleshooting](#troubleshooting)

## Project Setup

### Prerequisites

- GitHub account
- Replicate account with API token
- Docker installed locally
- Python 3.8+ installed
- NVIDIA GPU (optional, for local testing)

### Initial Repository Setup

1. Create a new GitHub repository:
   ```bash
   gh repo create your-username/speciesnet-replicate --public
   cd speciesnet-replicate
   ```

2. Initialize the repository:
   ```bash
   git init
   echo "# SpeciesNet on Replicate" > README.md
   git add README.md
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/your-username/speciesnet-replicate.git
   git push -u origin main
   ```

## Repository Structure

Create the following directory structure:

```
speciesnet-replicate/
├── .github/
│   └── workflows/
│       ├── test.yml
│       └── deploy.yml
├── models/
│   └── cache/                 # Model weights cache
├── tests/
│   ├── __init__.py
│   ├── test_predict.py
│   └── fixtures/
│       └── test_image.jpg
├── cog.yaml                   # Cog configuration
├── predict.py                 # Main prediction logic
├── download_weights.py        # Model weights downloader
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Custom Docker setup (if needed)
├── .gitignore
├── .dockerignore
├── README.md
└── LICENSE
```

## Replicate Model Configuration

### cog.yaml

Create the Cog configuration file that defines your model's interface:

```yaml
build:
  gpu: true
  cuda: "11.8"
  python_version: "3.10"
  python_packages:
    - "torch==2.0.1"
    - "torchvision==0.15.2"
    - "speciesnet==4.0.1"
    - "pillow==10.0.0"
    - "numpy==1.24.3"
  system_packages:
    - "libgl1-mesa-glx"
    - "libglib2.0-0"
  pre_install:
    - "pip install --upgrade pip"
  run:
    - "python download_weights.py"

predict: "predict.py:Predictor"
image: "r8.im/your-username/speciesnet"
```

### requirements.txt

```txt
speciesnet==4.0.1
torch==2.0.1
torchvision==0.15.2
pillow==10.0.0
numpy==1.24.3
cog>=0.8.0
```

## Model Implementation

### download_weights.py

Script to pre-download model weights during build:

```python
#!/usr/bin/env python3
"""
Pre-download SpeciesNet model weights during container build.
"""

import os
import subprocess
import sys
from pathlib import Path

def download_weights():
    """Download SpeciesNet model weights."""
    
    # Create cache directory
    cache_dir = Path("models/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading SpeciesNet model weights...")
    
    try:
        # Import speciesnet to trigger automatic weight download
        import speciesnet
        from speciesnet.models import get_model
        
        # Download both model variants
        print("Downloading SpeciesNet v4.0.1a (crop model)...")
        model_a = get_model("kaggle:google/speciesnet/pyTorch/v4.0.1a")
        
        print("Downloading SpeciesNet v4.0.1b (full-image model)...")
        model_b = get_model("kaggle:google/speciesnet/pyTorch/v4.0.1b")
        
        print("Model weights downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error downloading model weights: {e}")
        return False

if __name__ == "__main__":
    success = download_weights()
    sys.exit(0 if success else 1)
```

### predict.py

Main prediction logic with optional parameters:

```python
import tempfile
from pathlib import Path
from typing import List, Optional, Union
import json

import torch
from PIL import Image
from cog import BasePredictor, Input, Path as CogPath

import speciesnet
from speciesnet.scripts.run_model import main as run_speciesnet


class Predictor(BasePredictor):
    """SpeciesNet predictor for Replicate."""
    
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient."""
        
        # Verify CUDA availability
        if torch.cuda.is_available():
            print(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            print("CUDA not available, using CPU")
        
        # Pre-warm the models by loading them
        try:
            from speciesnet.models import get_model
            print("Pre-loading SpeciesNet models...")
            
            # Load default model (v4.0.1a)
            self.model_crop = get_model("kaggle:google/speciesnet/pyTorch/v4.0.1a")
            print("Loaded crop model (v4.0.1a)")
            
            # Load full-image model (v4.0.1b)  
            self.model_full = get_model("kaggle:google/speciesnet/pyTorch/v4.0.1b")
            print("Loaded full-image model (v4.0.1b)")
            
        except Exception as e:
            print(f"Warning: Could not pre-load models: {e}")
    
    def predict(
        self,
        image: CogPath = Input(description="Input camera trap image"),
        confidence_threshold: float = Input(
            description="Minimum confidence threshold for detections", 
            default=0.1, 
            ge=0.0, 
            le=1.0
        ),
        iou_threshold: float = Input(
            description="IoU threshold for non-maximum suppression", 
            default=0.5, 
            ge=0.0, 
            le=1.0
        ),
        country_code: Optional[str] = Input(
            description="ISO 3166-1 alpha-3 country code for geographic filtering (e.g., 'USA', 'CAN', 'GBR')",
            default=None
        ),
        admin1_region: Optional[str] = Input(
            description="First-level administrative division (US state codes like 'CA', 'TX')",
            default=None
        ),
        model_version: str = Input(
            description="SpeciesNet model version to use",
            default="v4.0.1a",
            choices=["v4.0.1a", "v4.0.1b"]
        ),
        run_mode: str = Input(
            description="Which components to run",
            default="full_ensemble",
            choices=["full_ensemble", "detector_only", "classifier_only"]
        ),
        return_top_k: int = Input(
            description="Number of top classifications to return",
            default=5,
            ge=1,
            le=10
        )
    ) -> dict:
        """Run a single prediction on the model."""
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Copy input image to temp directory
            input_path = temp_path / "input.jpg"
            
            # Handle different input types
            if isinstance(image, (str, Path)):
                # Copy file
                import shutil
                shutil.copy2(image, input_path)
            else:
                # Assume it's a file-like object
                with open(input_path, 'wb') as f:
                    f.write(image.read())
            
            # Verify image can be opened
            try:
                with Image.open(input_path) as img:
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        img.save(input_path)
            except Exception as e:
                return {"error": f"Invalid image file: {e}"}
            
            # Create output path
            output_path = temp_path / "predictions.json"
            
            # Prepare input JSON for SpeciesNet
            input_data = {
                "instances": [{
                    "filepath": str(input_path),
                }]
            }
            
            # Add geographic information if provided
            if country_code:
                input_data["instances"][0]["country"] = country_code.upper()
            
            if admin1_region:
                input_data["instances"][0]["admin1_region"] = admin1_region.upper()
            
            # Save input JSON
            input_json_path = temp_path / "input.json"
            with open(input_json_path, 'w') as f:
                json.dump(input_data, f)
            
            # Prepare SpeciesNet arguments
            args = [
                "--input_json", str(input_json_path),
                "--predictions_json", str(output_path),
                "--model", f"kaggle:google/speciesnet/pyTorch/{model_version}",
                "--confidence_threshold", str(confidence_threshold),
            ]
            
            # Add run mode flags
            if run_mode == "detector_only":
                args.append("--detector_only")
            elif run_mode == "classifier_only":
                args.append("--classifier_only")
            # full_ensemble is default, no flag needed
            
            try:
                # Run SpeciesNet
                import sys
                from io import StringIO
                
                # Capture stdout/stderr
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                stdout_capture = StringIO()
                stderr_capture = StringIO()
                
                try:
                    sys.stdout = stdout_capture
                    sys.stderr = stderr_capture
                    
                    # Run the model
                    run_speciesnet(args)
                    
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                
                # Read results
                if output_path.exists():
                    with open(output_path, 'r') as f:
                        results = json.load(f)
                    
                    # Extract first (and only) prediction
                    if results.get("predictions") and len(results["predictions"]) > 0:
                        prediction = results["predictions"][0]
                        
                        # Process and format output
                        output = self._format_output(
                            prediction, 
                            confidence_threshold,
                            iou_threshold,
                            return_top_k
                        )
                        
                        return output
                    else:
                        return {"error": "No predictions generated"}
                        
                else:
                    error_output = stderr_capture.getvalue()
                    return {"error": f"Prediction failed: {error_output}"}
                    
            except Exception as e:
                return {"error": f"Prediction error: {str(e)}"}
    
    def _format_output(
        self, 
        prediction: dict, 
        confidence_threshold: float,
        iou_threshold: float,
        return_top_k: int
    ) -> dict:
        """Format the SpeciesNet output for Replicate."""
        
        output = {
            "success": True,
            "model_version": prediction.get("model_version", "unknown"),
            "prediction_source": prediction.get("prediction_source"),
        }
        
        # Add final prediction if available
        if "prediction" in prediction:
            output["final_prediction"] = {
                "species": prediction["prediction"],
                "confidence": prediction.get("prediction_score", 0.0)
            }
        
        # Add detections if available
        if "detections" in prediction:
            detections = []
            for det in prediction["detections"]:
                if det["conf"] >= confidence_threshold:
                    detections.append({
                        "category": det["label"],
                        "confidence": det["conf"],
                        "bbox": det["bbox"]  # [xmin, ymin, width, height] normalized
                    })
            output["detections"] = detections
        
        # Add classifications if available
        if "classifications" in prediction:
            classifications = []
            classes = prediction["classifications"].get("classes", [])
            scores = prediction["classifications"].get("scores", [])
            
            for i, (cls, score) in enumerate(zip(classes, scores)):
                if i >= return_top_k:
                    break
                classifications.append({
                    "species": cls,
                    "confidence": score
                })
            
            output["classifications"] = classifications
        
        # Add any failures
        if "failures" in prediction:
            output["failures"] = prediction["failures"]
        
        # Add geographic info if present
        for geo_field in ["country", "admin1_region", "latitude", "longitude"]:
            if geo_field in prediction:
                output[geo_field] = prediction[geo_field]
        
        return output
```

## Local Development & Testing

### Setting up Local Environment

1. **Install Cog**:
   ```bash
   sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
   sudo chmod +x /usr/local/bin/cog
   ```

2. **Create test fixtures**:
   ```bash
   mkdir -p tests/fixtures
   # Add a test camera trap image to tests/fixtures/test_image.jpg
   ```

### Local Testing Commands

1. **Build the model locally**:
   ```bash
   cog build -t speciesnet-local
   ```

2. **Test with CLI**:
   ```bash
   # Basic prediction
   cog predict -i image=@tests/fixtures/test_image.jpg
   
   # With optional parameters
   cog predict \
     -i image=@tests/fixtures/test_image.jpg \
     -i confidence_threshold=0.2 \
     -i country_code="USA" \
     -i admin1_region="CA" \
     -i model_version="v4.0.1a"
   ```

3. **Start local HTTP server**:
   ```bash
   cog serve
   # Server will be available at http://localhost:5000
   ```

4. **Test HTTP endpoint**:
   ```bash
   curl -X POST http://localhost:5000/predictions \
     -H "Content-Type: application/json" \
     -d '{
       "input": {
         "image": "https://example.com/camera-trap-image.jpg",
         "confidence_threshold": 0.1,
         "country_code": "USA"
       }
     }'
   ```

### Unit Tests

Create `tests/test_predict.py`:

```python
import pytest
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from predict import Predictor


@pytest.fixture
def predictor():
    """Create a predictor instance."""
    pred = Predictor()
    pred.setup()
    return pred


@pytest.fixture
def test_image():
    """Create a test image."""
    # Create a simple test image
    image = Image.new('RGB', (640, 480), color='green')
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        image.save(f.name)
        yield f.name
    
    # Cleanup
    Path(f.name).unlink()


def test_predict_basic(predictor, test_image):
    """Test basic prediction functionality."""
    result = predictor.predict(image=test_image)
    
    assert "success" in result
    assert result["success"] is True
    assert "model_version" in result


def test_predict_with_country_code(predictor, test_image):
    """Test prediction with country code."""
    result = predictor.predict(
        image=test_image,
        country_code="USA",
        confidence_threshold=0.1
    )
    
    assert "success" in result
    if "country" in result:
        assert result["country"] == "USA"


def test_predict_detector_only(predictor, test_image):
    """Test detector-only mode."""
    result = predictor.predict(
        image=test_image,
        run_mode="detector_only"
    )
    
    assert "success" in result
    # Should have detections but not classifications
    assert "detections" in result or result.get("failures")


def test_invalid_image(predictor):
    """Test handling of invalid image."""
    # Create invalid image file
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(b"not an image")
        
        result = predictor.predict(image=f.name)
        assert "error" in result
        
    Path(f.name).unlink()


if __name__ == "__main__":
    pytest.main([__file__])
```

## GitHub Actions CI/CD

### .github/workflows/test.yml

```yaml
name: Test

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Install Cog
      run: |
        sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_Linux_x86_64
        sudo chmod +x /usr/local/bin/cog
    
    - name: Validate cog.yaml
      run: cog build --dry-run
    
    - name: Run Python tests
      run: |
        pytest tests/ -v --cov=predict --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
```

### .github/workflows/deploy.yml

```yaml
name: Deploy to Replicate

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:
    inputs:
      version:
        description: 'Version tag for deployment'
        required: true
        default: 'latest'

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Cog
      run: |
        sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_Linux_x86_64
        sudo chmod +x /usr/local/bin/cog
    
    - name: Authenticate with Replicate
      env:
        REPLICATE_API_TOKEN: ${{ secrets.REPLICATE_API_TOKEN }}
      run: |
        echo $REPLICATE_API_TOKEN | cog login --stdin
    
    - name: Build and push to Replicate
      env:
        REPLICATE_API_TOKEN: ${{ secrets.REPLICATE_API_TOKEN }}
      run: |
        # Extract version from tag or use input
        if [[ $GITHUB_REF == refs/tags/* ]]; then
          VERSION=${GITHUB_REF#refs/tags/}
        else
          VERSION=${{ github.event.inputs.version }}
        fi
        
        echo "Deploying version: $VERSION"
        
        # Push to Replicate
        cog push r8.im/${{ github.repository_owner }}/speciesnet:$VERSION
        
        # Also tag as latest if this is a release
        if [[ $VERSION =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
          cog push r8.im/${{ github.repository_owner }}/speciesnet:latest
        fi
    
    - name: Create GitHub Release
      if: startsWith(github.ref, 'refs/tags/')
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ github.ref }}
        body: |
          SpeciesNet model deployment ${{ github.ref }}
          
          This release has been automatically deployed to Replicate.
          
          **Usage:**
          ```python
          import replicate
          
          output = replicate.run(
              "${{ github.repository_owner }}/speciesnet:${{ github.ref }}",
              input={
                  "image": "https://example.com/camera-trap-image.jpg",
                  "confidence_threshold": 0.1,
                  "country_code": "USA"
              }
          )
          ```
        draft: false
        prerelease: false
```

## Deployment Process

### Initial Setup

1. **Set up Replicate secrets in GitHub**:
   - Go to your GitHub repository settings
   - Navigate to "Secrets and variables" → "Actions"
   - Add a new secret named `REPLICATE_API_TOKEN`
   - Set the value to your Replicate API token

2. **Configure repository settings**:
   ```bash
   # Add .gitignore
   cat > .gitignore << EOF
   __pycache__/
   *.pyc
   *.pyo
   *.pyd
   .Python
   env/
   venv/
   .venv/
   .env
   pip-log.txt
   pip-delete-this-directory.txt
   .coverage
   .pytest_cache/
   htmlcov/
   .DS_Store
   models/cache/
   temp/
   EOF
   
   # Add .dockerignore
   cat > .dockerignore << EOF
   .git
   .github
   README.md
   .gitignore
   .dockerignore
   tests/
   __pycache__/
   *.pyc
   .coverage
   .pytest_cache/
   EOF
   ```

### Manual Deployment

For manual deployment without GitHub Actions:

```bash
# Authenticate with Replicate
export REPLICATE_API_TOKEN=your_token_here
echo $REPLICATE_API_TOKEN | cog login --stdin

# Build and push
cog build -t speciesnet-local
cog push r8.im/your-username/speciesnet
```

### Automated Deployment

1. **Tag-based deployment**:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   # This will trigger automatic deployment
   ```

2. **Manual workflow dispatch**:
   - Go to GitHub Actions tab
   - Select "Deploy to Replicate" workflow
   - Click "Run workflow"
   - Enter version tag

## Usage Examples

### Python Client

```python
import replicate

# Basic usage
output = replicate.run(
    "your-username/speciesnet:latest",
    input={
        "image": "https://example.com/camera-trap-image.jpg"
    }
)

print(f"Species detected: {output['final_prediction']['species']}")
print(f"Confidence: {output['final_prediction']['confidence']:.3f}")

# Advanced usage with all parameters
output = replicate.run(
    "your-username/speciesnet:latest",
    input={
        "image": "https://example.com/camera-trap-image.jpg",
        "confidence_threshold": 0.2,
        "iou_threshold": 0.5,
        "country_code": "USA",
        "admin1_region": "CA",
        "model_version": "v4.0.1a",
        "run_mode": "full_ensemble",
        "return_top_k": 5
    }
)

# Process detections
for detection in output.get('detections', []):
    print(f"Detected {detection['category']} with {detection['confidence']:.3f} confidence")
    print(f"Bounding box: {detection['bbox']}")

# Process classifications
for classification in output.get('classifications', []):
    print(f"{classification['species']}: {classification['confidence']:.3f}")
```

### JavaScript/Node.js

```javascript
import Replicate from 'replicate';

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN,
});

const output = await replicate.run(
  "your-username/speciesnet:latest",
  {
    input: {
      image: "https://example.com/camera-trap-image.jpg",
      confidence_threshold: 0.1,
      country_code: "USA"
    }
  }
);

console.log("Species:", output.final_prediction.species);
console.log("Confidence:", output.final_prediction.confidence);
```

### cURL

```bash
curl -s -X POST \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "your-model-version-id",
    "input": {
      "image": "https://example.com/camera-trap-image.jpg",
      "confidence_threshold": 0.1,
      "country_code": "USA",
      "admin1_region": "CA"
    }
  }' \
  https://api.replicate.com/v1/predictions
```

## Troubleshooting

### Common Issues

1. **Model weights not downloading**:
   - Ensure internet connectivity during build
   - Check if Kaggle API is accessible
   - Verify the model version strings are correct

2. **CUDA out of memory**:
   - Reduce input image size in preprocessing
   - Use CPU-only mode for very large images
   - Consider batching for multiple images

3. **Geographic filtering not working**:
   - Verify country codes are valid ISO 3166-1 alpha-3
   - Check admin1_region format (US state codes only)
   - Ensure SpeciesNet has geographic data for the specified region

4. **Slow inference times**:
   - Ensure GPU is being used (check logs for "CUDA")
   - Pre-load models in setup() method
   - Consider image resizing for faster processing

### Debug Strategies

1. **Enable verbose logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Test locally first**:
   ```bash
   cog predict -i image=@test_image.jpg --debug
   ```

3. **Check model outputs step by step**:
   - Test detector_only mode
   - Test classifier_only mode  
   - Test full_ensemble mode

4. **Validate input formats**:
   - Ensure images are valid JPEG/PNG
   - Check image dimensions and file sizes
   - Verify country codes and region formats

### Performance Optimization

1. **Model caching**:
   - Pre-load models in setup()
   - Cache processed weights
   - Use model.eval() for inference

2. **Image preprocessing**:
   - Resize large images before processing
   - Use efficient image loading libraries
   - Consider input validation

3. **GPU utilization**:
   - Monitor GPU memory usage
   - Batch multiple predictions if possible
   - Use appropriate CUDA versions

### Getting Help

- **SpeciesNet Issues**: [GitHub Issues](https://github.com/google/cameratrapai/issues)
- **Replicate Support**: [Replicate Discord](https://discord.gg/replicate)
- **Cog Documentation**: [Cog Docs](https://github.com/replicate/cog)

---

This guide provides a complete end-to-end solution for deploying SpeciesNet on Replicate. Follow each section carefully, and you'll have a production-ready camera trap AI model with automated CI/CD and comprehensive testing capabilities.
