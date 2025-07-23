# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **SpeciesNet Replicate Deployment project** designed to wrap Google's SpeciesNet camera trap AI model and deploy it on Replicate with automated CI/CD. The project is currently in the **planning/documentation phase** - it contains comprehensive implementation guides but no actual code yet.

**Key Technologies:**
- **ML Framework**: PyTorch, SpeciesNet (Google's camera trap AI)  
- **Deployment**: Replicate platform using Cog
- **Python Version**: 3.10 with CUDA 11.8 support
- **CI/CD**: GitHub Actions
- **Testing**: pytest with local Cog testing

## Project Status

🚨 **IMPORTANT**: This repository contains only documentation and empty configuration files. The actual implementation needs to be built according to the detailed specifications in `GUIDE.md`.

**Current State:**
- ✅ Comprehensive deployment guide (`GUIDE.md`)
- ❌ No implementation code exists
- ❌ No working configuration files  
- ❌ No CI/CD workflows implemented

## Architecture

The intended architecture (from GUIDE.md) combines multiple AI models:
- **MegaDetector**: Object detection for animals, humans, vehicles
- **SpeciesNet Classifier**: Species classification for 2000+ labels
- **Geographic Filtering**: Country/region-based filtering
- **Model Variants**: v4.0.1a (crop) and v4.0.1b (full-image)

## Key Files to Implement

Based on `GUIDE.md`, these are the critical files that need to be created:

### Core Implementation
- `predict.py` - Main Replicate predictor class with SpeciesNet integration
- `download_weights.py` - Pre-download model weights during container build
- `cog.yaml` - Replicate/Cog configuration (GPU, CUDA 11.8, Python 3.10)
- `requirements.txt` - Python dependencies

### Development & Testing
- `tests/test_predict.py` - Unit tests for prediction functionality
- `tests/fixtures/test_image.jpg` - Test camera trap image

### CI/CD Workflows
- `.github/workflows/test.yml` - Automated testing pipeline
- `.github/workflows/deploy.yml` - Replicate deployment automation

## Development Commands

### Local Development (once implemented)
```bash
# Install Cog (Replicate's deployment tool)
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
sudo chmod +x /usr/local/bin/cog

# Build model locally
cog build -t speciesnet-local

# Test prediction
cog predict -i image=@tests/fixtures/test_image.jpg

# Test with parameters
cog predict -i image=@tests/fixtures/test_image.jpg -i confidence_threshold=0.2 -i country_code="USA"

# Start local HTTP server
cog serve
```

### Testing
```bash
# Run unit tests
pytest tests/ -v --cov=predict

# Validate Cog configuration
cog build --dry-run
```

### Deployment
```bash
# Manual deployment
cog push r8.im/username/speciesnet

# Automated via git tags
git tag v1.0.0
git push origin v1.0.0
```

## Model Configuration

The SpeciesNet model supports these key parameters:
- `confidence_threshold`: Detection confidence (0.0-1.0, default: 0.1)
- `country_code`: ISO 3166-1 alpha-3 codes ("USA", "CAN", "GBR")
- `admin1_region`: US state codes ("CA", "TX")
- `model_version`: "v4.0.1a" (crop) or "v4.0.1b" (full-image)
- `run_mode`: "full_ensemble", "detector_only", "classifier_only"

## Important Implementation Notes

### Security & Best Practices
- Never commit API tokens or secrets to the repository
- Use GitHub Secrets for `REPLICATE_API_TOKEN`
- Validate all input images before processing
- Handle errors gracefully with proper error messages

### Model Performance
- Pre-load models in `setup()` method for efficiency
- Use CUDA when available, fall back to CPU
- Cache model weights during container build
- Validate GPU memory usage for large images

### Testing Strategy
- Test locally with Cog before deployment
- Create meaningful unit tests for edge cases
- Test both model variants (v4.0.1a and v4.0.1b)
- Validate geographic filtering functionality

## Development Workflow

1. **Start by implementing core files** according to `GUIDE.md` specifications
2. **Test locally** using Cog CLI before pushing to Replicate  
3. **Set up CI/CD** only after core functionality works
4. **Follow the exact specifications** in `GUIDE.md` - it contains production-ready code templates

## Getting Started

When implementing this project:

1. **Read `GUIDE.md` thoroughly** - it contains complete, production-ready code templates
2. **Start with `predict.py`** - the core predictor class
3. **Create `cog.yaml`** with the exact specifications from the guide
4. **Test locally** before setting up automation
5. **Follow the directory structure** exactly as specified in the guide

The `GUIDE.md` file contains comprehensive, copy-paste ready implementation code - use it as your primary reference for all implementation decisions.