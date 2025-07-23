import tempfile
from pathlib import Path
from typing import List, Optional, Union
import json
import os
import sys

import torch
from PIL import Image
from cog import BasePredictor, Input, Path as CogPath

# Import speciesnet only when needed to avoid setup failures
# import speciesnet
# from speciesnet.scripts.run_model import main as run_speciesnet


class Predictor(BasePredictor):
    """SpeciesNet predictor for Replicate."""
    
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient."""
        
        # Try multiple logging approaches
        print("SETUP: Starting predictor setup", flush=True)
        sys.stderr.write("STDERR: Starting predictor setup\n")
        sys.stderr.flush()
        
        # Verify CUDA availability and force GPU usage
        if torch.cuda.is_available():
            print(f"SETUP: CUDA available with {torch.cuda.device_count()} GPUs", flush=True)
            sys.stderr.write(f"STDERR: CUDA available with {torch.cuda.device_count()} GPUs\n")
            print(f"SETUP: Using GPU: {torch.cuda.get_device_name()}", flush=True)
            sys.stderr.write(f"STDERR: Using GPU: {torch.cuda.get_device_name()}\n")
            print(f"SETUP: CUDA version: {torch.version.cuda}", flush=True)
            # Set CUDA device
            torch.cuda.set_device(0)
            print(f"SETUP: Active CUDA device: {torch.cuda.current_device()}", flush=True)
        else:
            print("SETUP: CUDA not available, using CPU", flush=True)
            sys.stderr.write("STDERR: CUDA not available, using CPU\n")
        
        sys.stderr.flush()
        print("SETUP: Model setup complete", flush=True)
    
    def _load_model_if_needed(self, model_version: str):
        """Load model on demand if not already loaded."""
        # SpeciesNet 5.0.0 is used via command-line interface, models load automatically
        # This method is kept for compatibility but doesn't need to preload models
        print(f"SpeciesNet will load model {model_version} automatically when needed")
    
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
        
        print("PREDICT: Starting prediction")
        print(f"PREDICT: Model version: {model_version}, Run mode: {run_mode}")
        
        # Load model if needed
        self._load_model_if_needed(model_version)
        
        print("PREDICT: Creating temporary directory")
        
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
            
            # Use SpeciesNet 5.0.0 command-line interface via subprocess
            import subprocess
            
            # Prepare SpeciesNet command
            cmd = [
                "python", "-m", "speciesnet.scripts.run_model",
                "--instances_json", str(input_json_path),
                "--predictions_json", str(output_path),
                "--model", f"kaggle:google/speciesnet/pyTorch/{model_version}",
            ]
            
            # Add run mode flags
            if run_mode == "detector_only":
                cmd.append("--detector_only")
            elif run_mode == "classifier_only":
                cmd.append("--classifier_only")
            # full_ensemble is default, no flag needed
            
            try:
                print("PREDICT: About to run SpeciesNet")
                
                # Log GPU status before running SpeciesNet
                if torch.cuda.is_available():
                    print(f"PREDICT: GPU Memory before SpeciesNet: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
                    print(f"PREDICT: GPU Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
                else:
                    print("PREDICT: No CUDA available for SpeciesNet")
                
                print(f"PREDICT: Command: {' '.join(cmd)}")
                
                # Set environment variables to bypass interactive prompts and force GPU
                env = os.environ.copy()
                env.update({
                    'PYTHONUNBUFFERED': '1',  # Prevent output buffering
                    'SPECIESNET_AUTO_CONFIRM': '1',  # Potential auto-confirm flag
                    'NO_INTERACTIVE': '1',  # Common bypass flag
                    'CUDA_VISIBLE_DEVICES': '0',  # Ensure GPU 0 is visible
                    'TORCH_USE_CUDA_DSA': '1',  # Force CUDA usage
                    'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',  # Optimize memory
                })
                
                print("PREDICT: Running SpeciesNet subprocess")
                
                # Run SpeciesNet via subprocess with stdin bypass
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                    check=False,
                    input="y\ny\ny\ny\n",  # Auto-answer "yes" to multiple prompts
                    env=env
                )
                
                print(f"PREDICT: SpeciesNet completed with return code: {result.returncode}")
                
                if result.returncode != 0:
                    return {
                        "error": f"SpeciesNet failed: {result.stderr or 'Unknown error'}"
                    }
                
                print("PREDICT: Checking for output file")
                
                # Read results
                if output_path.exists():
                    print("PREDICT: Output file found, reading results")
                    with open(output_path, 'r') as f:
                        results = json.load(f)
                    
                    # Extract first (and only) prediction
                    if results.get("predictions") and len(results["predictions"]) > 0:
                        prediction = results["predictions"][0]
                        
                        print("PREDICT: Processing and formatting output")
                        
                        # Process and format output
                        output = self._format_output(
                            prediction, 
                            confidence_threshold,
                            iou_threshold,
                            return_top_k
                        )
                        
                        print("PREDICT: Prediction completed successfully")
                        return output
                    else:
                        print("PREDICT: No predictions found in results")
                        return {"error": "No predictions generated"}
                        
                else:
                    print("PREDICT: Output file not found")
                    return {
                        "error": f"Prediction failed: {result.stderr or 'No output file generated'}"
                    }
                    
            except subprocess.TimeoutExpired:
                return {"error": "SpeciesNet prediction timed out after 5 minutes"}
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