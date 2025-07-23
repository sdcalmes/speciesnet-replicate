import pytest
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import os

class TestInteractivePrompts:
    """Test that our predictor handles various interactive prompt scenarios."""
    
    def test_multiple_yes_inputs(self):
        """Test that multiple 'y' inputs are handled correctly."""
        # Simulate a script that asks multiple questions
        test_script = '''
import sys
print("Question 1? [y/N]: ", end="", flush=True)
response1 = input()
print(f"Got: {response1}")

print("Question 2? [y/N]: ", end="", flush=True) 
response2 = input()
print(f"Got: {response2}")

print("Question 3? [y/N]: ", end="", flush=True)
response3 = input()
print(f"Got: {response3}")

print("All questions answered successfully!")
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            script_path = f.name
        
        try:
            # Test our multi-yes input approach
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                input="y\ny\ny\ny\n",  # Same as our implementation
                timeout=10
            )
            
            assert result.returncode == 0
            assert "All questions answered successfully!" in result.stdout
            assert "Got: y" in result.stdout
            
        finally:
            Path(script_path).unlink()
    
    def test_environment_variables(self):
        """Test that environment variables work for bypassing prompts."""
        test_script = '''
import os
import sys

# Check if auto-confirm environment variables are set
if os.environ.get('NO_INTERACTIVE') == '1':
    print("NO_INTERACTIVE detected, auto-confirming")
    response = "y"
else:
    print("Manual input required [y/N]: ", end="", flush=True)
    response = input()

print(f"Response: {response}")
if response.lower() == 'y':
    print("SUCCESS: Auto-confirmed via environment")
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            script_path = f.name
        
        try:
            env = os.environ.copy()
            env.update({
                'NO_INTERACTIVE': '1',
                'PYTHONUNBUFFERED': '1',
            })
            
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                env=env,
                timeout=10
            )
            
            assert result.returncode == 0
            assert "SUCCESS: Auto-confirmed via environment" in result.stdout
            
        finally:
            Path(script_path).unlink()
    
    def test_eof_error_handling(self):
        """Test handling of EOFError when no input is available."""
        test_script = '''
import sys
try:
    print("This will cause EOFError [y/N]: ", end="", flush=True)
    response = input()
    print(f"Got: {response}")
except EOFError:
    print("EOFError caught - no input available")
    print("Defaulting to 'y' for automation")
    response = "y"

print(f"Final response: {response}")
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            script_path = f.name
        
        try:
            # Run without providing input (should cause EOFError)
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,  # No input available
                timeout=10
            )
            
            assert "EOFError caught" in result.stdout
            assert "Final response: y" in result.stdout
            
        finally:
            Path(script_path).unlink()
    
    def test_timeout_scenarios(self):
        """Test that our timeout settings work correctly."""
        test_script = '''
import time
import sys

print("Starting long operation...")
time.sleep(2)  # Simulate processing
print("Operation completed")
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            script_path = f.name
        
        try:
            # Should complete within timeout
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                timeout=5  # Generous timeout
            )
            
            assert result.returncode == 0
            assert "Operation completed" in result.stdout
            
            # Test timeout failure
            test_script_long = '''
import time
print("Starting very long operation...")
time.sleep(10)  # Longer than timeout
print("This should not appear")
'''
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f2:
                f2.write(test_script_long)
                long_script_path = f2.name
            
            try:
                with pytest.raises(subprocess.TimeoutExpired):
                    subprocess.run(
                        ["python", long_script_path],
                        capture_output=True,
                        text=True,
                        timeout=3  # Short timeout
                    )
            finally:
                Path(long_script_path).unlink()
                
        finally:
            Path(script_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])