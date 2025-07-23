import pytest
import tempfile
import time
from pathlib import Path
from PIL import Image
from unittest.mock import patch
import subprocess


class TestProductionPrompts:
    """Test our production predictor handles prompts correctly."""
    
    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        image = Image.new('RGB', (640, 480), color='green')
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            image.save(f.name)
            yield f.name
        
        Path(f.name).unlink()
    
    def test_predictor_prompt_handling(self, test_image):
        """Test that our production predictor handles prompts correctly."""
        # Import our test predictor (avoids Cog dependency issues)
        from test_predict import TestPredictor
        
        predictor = TestPredictor()
        predictor.setup()
        
        # Test all run modes
        modes = ["full_ensemble", "detector_only", "classifier_only"]
        
        for mode in modes:
            start_time = time.time()
            
            result = predictor.predict(
                image=test_image,
                run_mode=mode
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Should complete quickly (our test predictor is mocked)
            assert duration < 1, f"Test predictor took {duration}s for {mode}"
            assert result["success"] is True
            
            print(f"✅ {mode}: {duration:.3f}s")
    
    def test_subprocess_prompt_protection(self):
        """Test our subprocess call pattern against various prompt scenarios."""
        
        # Test script that simulates different types of prompts
        prompt_scenarios = [
            # Standard y/N prompt
            'print("Continue? [y/N]: ", end="", flush=True); response = input(); print(f"Got: {response}")',
            
            # Multiple prompts
            'print("Q1 [y/N]: ", end="", flush=True); r1 = input(); print("Q2 [y/N]: ", end="", flush=True); r2 = input(); print(f"Got: {r1}, {r2}")',
            
            # Different prompt formats
            'print("Proceed (yes/no)? ", end="", flush=True); response = input(); print(f"Got: {response}")',
            
            # License acceptance style
            'print("Accept terms? [Y/n]: ", end="", flush=True); response = input(); print(f"Got: {response}")',
        ]
        
        for i, scenario in enumerate(prompt_scenarios):
            test_script = f'''
import sys
try:
    {scenario}
    print("SUCCESS: Prompt handled")
except EOFError:
    print("ERROR: EOFError - prompt not handled")
    sys.exit(1)
'''
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                script_path = f.name
            
            try:
                # Use our production approach
                import os
                env = os.environ.copy()
                env.update({
                    'PYTHONUNBUFFERED': '1',
                    'SPECIESNET_AUTO_CONFIRM': '1',
                    'NO_INTERACTIVE': '1',
                })
                
                result = subprocess.run(
                    ["python", script_path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    input="y\ny\ny\ny\n",  # Our multi-yes approach
                    env=env,
                    check=False
                )
                
                assert result.returncode == 0, f"Scenario {i+1} failed: {result.stderr}"
                assert "SUCCESS: Prompt handled" in result.stdout
                assert "ERROR: EOFError" not in result.stdout
                
                print(f"✅ Scenario {i+1}: Handled correctly")
                
            finally:
                Path(script_path).unlink()
    
    def test_timeout_protection(self):
        """Test that our timeout settings protect against hanging prompts."""
        
        # Script that would hang waiting for input
        hanging_script = '''
import time
print("About to hang waiting for input...")
try:
    response = input("This will hang [y/N]: ")
    print(f"Got unexpected response: {response}")
except EOFError:
    print("Protected by EOFError handling")
except KeyboardInterrupt:
    print("Protected by timeout/interrupt")
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(hanging_script)
            script_path = f.name
        
        try:
            start_time = time.time()
            
            # This should NOT hang due to our input provision
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                timeout=5,  # Short timeout for safety
                input="y\n",
                check=False
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Should complete quickly, not hang
            assert duration < 3, f"Script took {duration}s, likely hung waiting for input"
            
            print(f"✅ Timeout protection: Completed in {duration:.3f}s")
            
        finally:
            Path(script_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])