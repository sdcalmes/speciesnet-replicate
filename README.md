# SpeciesNet on Replicate

A complete deployment of Google's SpeciesNet camera trap AI model on Replicate with automated CI/CD.

## Overview

SpeciesNet is an ensemble of AI models for classifying wildlife in camera trap images. This deployment combines:
- **MegaDetector**: Object detection for animals, humans, and vehicles
- **SpeciesNet Classifier**: Species-level classification for 2000+ labels
- **Geographic Filtering**: Country/region-based filtering for improved accuracy

## Features

- 🚀 **Production-ready deployment** on Replicate
- 🔧 **Flexible parameters** for confidence thresholds, geographic filtering
- 📊 **Two model variants**: crop model (v4.0.1a) and full-image model (v4.0.1b)
- 🧪 **Complete testing suite** with pytest
- 🔄 **Automated CI/CD** with GitHub Actions
- 📦 **Docker containerization** with Cog

## Quick Start

### Using the Model

```python
import replicate

output = replicate.run(
    "your-username/speciesnet:latest",
    input={
        "image": "https://example.com/camera-trap-image.jpg",
        "confidence_threshold": 0.1,
        "country_code": "USA",
        "admin1_region": "CA"
    }
)

print(f"Species: {output['final_prediction']['species']}")
print(f"Confidence: {output['final_prediction']['confidence']:.3f}")
```

### Local Development

1. **Install Cog**:
   ```bash
   sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
   sudo chmod +x /usr/local/bin/cog
   ```

2. **Build and test locally**:
   ```bash
   # Standard build (models download during build)
   cog build -t speciesnet-local
   cog predict -i image=@tests/fixtures/test_image.jpg
   
   # Fast build with pre-downloaded weights
   cog build --separate-weights -t speciesnet-fast
   ```

3. **Run tests**:
   ```bash
   pip install pytest pytest-cov
   pytest tests/ -v
   ```

## Performance Optimizations

The deployment includes several performance optimizations:

### **Embedded Model Weights (Recommended)**
Following Cog best practices, model weights are embedded directly in the Docker image:
- **Faster setup()**: Near-instant startup (~5 seconds vs 5+ minutes)
- **Self-contained**: No external dependencies during inference
- **Reproducible**: Models are version-locked in the image
- **Reliable**: No network issues during cold starts

### **Build Commands**
```bash
# Production build with embedded weights (recommended)
cog build --separate-weights -t speciesnet-production

# Development build  
cog build -t speciesnet-local
```

### **Performance Comparison**

| Method | Build Time | Setup Time | Cold Start | Reliability |
|--------|------------|------------|------------|-------------|
| **Embedded Weights** | ~15 min | ~5 sec | Fast | High |
| Download in setup() | ~5 min | ~5 min | Slow | Network dependent |

### **Expected Performance**
- **Setup time**: ~5 seconds (vs 5+ minutes)
- **First prediction**: ~20 seconds  
- **Subsequent predictions**: ~20 seconds
- **Build time**: ~15 minutes (one-time cost for major speedup)

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | Image | Required | Input camera trap image |
| `confidence_threshold` | Float | 0.1 | Minimum confidence for detections (0.0-1.0) |
| `iou_threshold` | Float | 0.5 | IoU threshold for NMS (0.0-1.0) |
| `country_code` | String | None | ISO 3166-1 alpha-3 country code |
| `admin1_region` | String | None | US state codes (CA, TX, etc.) |
| `model_version` | String | v4.0.1a | Model variant (v4.0.1a/v4.0.1b) |
| `run_mode` | String | full_ensemble | Mode (full_ensemble/detector_only/classifier_only) |
| `return_top_k` | Integer | 5 | Number of top classifications to return |

## Output Format

```json
{
  "success": true,
  "model_version": "v4.0.1a",
  "final_prediction": {
    "species": "Odocoileus virginianus",
    "confidence": 0.92
  },
  "detections": [
    {
      "category": "animal",
      "confidence": 0.95,
      "bbox": [0.1, 0.2, 0.3, 0.4]
    }
  ],
  "classifications": [
    {"species": "Odocoileus virginianus", "confidence": 0.92},
    {"species": "Cervus canadensis", "confidence": 0.08}
  ]
}
```

## Deployment

### Automated Deployment

Push a version tag to trigger automatic deployment:

```bash
git tag v1.0.0
git push origin v1.0.0
```

### Manual Deployment

```bash
export REPLICATE_API_TOKEN=your_token_here
cog push r8.im/your-username/speciesnet
```

## Project Structure

```
speciesnet-replicate/
├── .github/workflows/     # CI/CD workflows
├── models/cache/          # Model weights cache
├── tests/                 # Test suite
├── cog.yaml              # Replicate configuration
├── predict.py            # Main prediction logic
├── download_weights.py   # Model downloader
├── requirements.txt      # Dependencies
└── README.md
```

## Development

See [GUIDE.md](GUIDE.md) for complete implementation details and [CLAUDE.md](CLAUDE.md) for Claude Code guidance.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

- **SpeciesNet Issues**: [GitHub Issues](https://github.com/google/cameratrapai/issues)
- **Replicate Support**: [Replicate Discord](https://discord.gg/replicate)
- **Cog Documentation**: [Cog Docs](https://github.com/replicate/cog)