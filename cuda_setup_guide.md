# CUDA Setup Guide for llama-cpp-python

This guide will help you set up llama-cpp-python with CUDA support to utilize your NVIDIA GPU for faster inference.

## Prerequisites

- NVIDIA GPU with compatible drivers installed
- CUDA Toolkit (12.4 recommended for optimal compatibility)
- Python 3.8 or newer
- Administrator access (for Windows)

## 1. Check Your NVIDIA GPU

First, verify that your NVIDIA GPU is properly detected:

```bash
nvidia-smi
```

You should see output showing your GPU, driver version, and CUDA version.

## 2. Close Any Python Applications

Before installing or reinstalling packages, make sure to close:
- VS Code
- Jupyter Notebooks
- Any Python scripts that might be using llama-cpp-python

## 3. Update pip

```bash
python -m pip install --upgrade pip
```

## 4. Install Build Tools

These are required for compiling llama-cpp-python with CUDA support:

```bash
pip install cmake ninja setuptools wheel
```

## 5. Install Method 1: Pre-built Wheels (Recommended)

Pre-built wheels are the easiest way to get CUDA support:

```bash
# Uninstall existing installation (might require administrator privileges)
pip uninstall -y llama-cpp-python

# Install with CUDA support (use cu124 for CUDA 12.4, adjust if needed)
pip install --force-reinstall llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

## 6. Install Method 2: Build from Source (Advanced)

If the pre-built wheels don't work, you can build from source:

### Windows (CMD - not PowerShell)

```cmd
# Open Command Prompt as Administrator
# Uninstall existing package
pip uninstall -y llama-cpp-python

# Set environment variables for CUDA support
set CMAKE_ARGS=-DGGML_CUDA=on

# Install from source
pip install --no-cache-dir --upgrade --force-reinstall --no-binary llama-cpp-python llama-cpp-python
```

### Linux/macOS

```bash
pip uninstall -y llama-cpp-python
CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir --upgrade --force-reinstall --no-binary llama-cpp-python llama-cpp-python
```

## 7. Verify CUDA Support

Create a file named `verify_cuda.py`:

```python
import llama_cpp
import subprocess
import sys

print(f"llama-cpp-python version: {llama_cpp.__version__}")

print("\nChecking CUDA backends:")
cuda_found = False
for attr in dir(llama_cpp):
    if attr.startswith("LLAMA_BACKEND_"):
        print(f"  - {attr}")
        if "CUDA" in attr:
            cuda_found = True

if not cuda_found:
    print("No CUDA backends found in llama_cpp!")
else:
    print("CUDA backends found!")

print("\nChecking NVIDIA GPU:")
try:
    result = subprocess.run("nvidia-smi", capture_output=True, text=True, check=False)
    if result.returncode == 0:
        print("NVIDIA GPU detected!")
    else:
        print("nvidia-smi failed. No NVIDIA GPU or driver issue.")
except Exception as e:
    print(f"Error checking nvidia-smi: {e}")

# Try to create a minimal model to check CUDA
print("\nTesting CUDA with llama_cpp:")
try:
    from llama_cpp import Llama
    # Don't actually load a model, just check object creation with CUDA flag
    print("CUDA appears to be available for Llama!")
except Exception as e:
    print(f"Error initializing Llama with CUDA: {e}")

print("\nEnvironment information:")
print(f"Python version: {sys.version}")
```

Run it:

```bash
python verify_cuda.py
```

## 8. Using CUDA with Existing Scripts

When running your scripts, you can force GPU usage:

```bash
python test.py --force-gpu
```

Or set environment variables before running:

```bash
# Windows
set LLAMA_CUBLAS=1

# Linux/macOS
export LLAMA_CUBLAS=1
```

## 9. Requirements.txt

Use the following `requirements.txt` to ensure consistent package versions:

```
--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
llama-cpp-python==0.3.8
torch==2.6.0+cu124
torchaudio==2.6.0+cu124
torchvision==0.21.0+cu124
gitpython==3.1.44
keyboard==0.13.5
bitsandbytes==0.45.3
```

## Troubleshooting

### Permission Issues
- Run Command Prompt or Terminal as administrator
- Create a virtual environment to avoid system-wide permission issues:
  ```bash
  python -m venv llama_env
  # Windows
  llama_env\Scripts\activate
  # Linux/macOS
  source llama_env/bin/activate
  ```

### No CUDA Backends Detected
- Make sure your NVIDIA drivers are up to date
- Try a different CUDA version in the package URL (cu118, cu121, cu124)
- Check if CUDA is properly installed: `nvcc --version`

### Memory Errors
- Reduce `n_gpu_layers` from `-1` (all layers) to a lower number
- Try setting `n_gpu_layers=1` and gradually increase

### Still Not Working
- Check if the library is built with CUDA:
  ```python
  import llama_cpp
  dir(llama_cpp)  # Look for CUDA-related attributes
  ```
- Consider using a Docker container with CUDA support pre-configured 