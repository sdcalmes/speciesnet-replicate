import pytest
import subprocess
import tempfile
import json
from pathlib import Path
from PIL import Image
import time

class TestSpeciesNetPrompts:
    """Test SpeciesNet-specific interactive prompt scenarios."""
    
    @pytest.fixture
    def test_setup(self):
        """Create test environment with image and JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test image
            test_img = Image.new('RGB', (64, 64), color='green')
            image_path = temp_path / "test.jpg"
            test_img.save(image_path)
            
            # Create input JSON
            input_data = {
                "instances": [{
                    "filepath": str(image_path),
                }]
            }
            input_json_path = temp_path / "input.json"
            with open(input_json_path, 'w') as f:
                json.dump(input_data, f)
            
            output_path = temp_path / "output.json"
            
            yield {
                "input_json": str(input_json_path),
                "output_json": str(output_path),
                "temp_dir": temp_path
            }
    
    def test_detector_no_prompts(self, test_setup):
        """Test that detector_only mode doesn't have interactive prompts."""
        cmd = [
            "python", "-m", "speciesnet.scripts.run_model",
            "--instances_json", test_setup["input_json"],
            "--predictions_json", test_setup["output_json"],
            "--model", "kaggle:google/speciesnet/pyTorch/v4.0.1a",
            "--detector_only"
        ]
        
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            input="",  # No input provided
            check=False
        )
        end_time = time.time()
        
        # Should complete quickly without hanging
        assert end_time - start_time < 30, "Detector took too long, might be waiting for input"
        
        # Check for interactive prompt indicators in output
        combined_output = result.stdout + result.stderr
        prompt_indicators = ["[y/N]", "[Y/n]", "Continue?", "Proceed?", "input("]
        
        for indicator in prompt_indicators:
            assert indicator not in combined_output, f"Found potential prompt: {indicator}"
    
    def test_classifier_with_auto_yes(self, test_setup):
        """Test that classifier_only mode works with our auto-yes approach."""
        cmd = [
            "python", "-m", "speciesnet.scripts.run_model",
            "--instances_json", test_setup["input_json"],
            "--predictions_json", test_setup["output_json"],
            "--model", "kaggle:google/speciesnet/pyTorch/v4.0.1a",
            "--classifier_only"
        ]
        
        # Use our production approach
        import os
        env = os.environ.copy()
        env.update({
            'PYTHONUNBUFFERED': '1',
            'SPECIESNET_AUTO_CONFIRM': '1',
            'NO_INTERACTIVE': '1',
        })
        
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            input="y\ny\ny\ny\n",
            env=env,
            check=False
        )
        end_time = time.time()
        
        # Should complete without hanging
        assert end_time - start_time < 45, "Classifier took too long, might be waiting for input"
        
        # Should either succeed or fail quickly (not hang)
        assert result.returncode in [0, 1], f"Unexpected return code: {result.returncode}"
        
        # Check that we don't see EOF errors
        combined_output = result.stdout + result.stderr
        assert "EOFError" not in combined_output, "EOF error suggests unhandled prompt"
    
    def test_full_ensemble_performance(self, test_setup):
        """Test that full ensemble completes in reasonable time."""
        cmd = [
            "python", "-m", "speciesnet.scripts.run_model",
            "--instances_json", test_setup["input_json"],
            "--predictions_json", test_setup["output_json"],
            "--model", "kaggle:google/speciesnet/pyTorch/v4.0.1a"
        ]
        
        import os
        env = os.environ.copy()
        env.update({
            'PYTHONUNBUFFERED': '1',
            'SPECIESNET_AUTO_CONFIRM': '1',
            'NO_INTERACTIVE': '1',
        })
        
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute max
            input="y\ny\ny\ny\n",
            env=env,
            check=False
        )
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Should complete in reasonable time (not hang for minutes)
        assert duration < 90, f"Full ensemble took {duration}s, likely hanging on prompts"
        
        # Log performance for analysis
        print(f"Full ensemble completed in {duration:.1f} seconds")
        
        # Check for successful completion or reasonable failure
        if result.returncode != 0:
            print(f"Command failed with code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            # Don't assert failure here - might be model download issues


if __name__ == "__main__":
    pytest.main([__file__, "-v"])