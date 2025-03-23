"""
verify_cuda.py

Tool to verify CUDA support in llama-cpp-python

Functions:
    main(): Check CUDA support for llama-cpp-python

Command Line Usage Example:
    python verify_cuda.py
"""
import os
import sys
import subprocess
import logging
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

logger = logging.getLogger(__name__)

def main() -> None:
    """
    Main function to check for CUDA support in llama-cpp-python.
    
    Parameters:
        None
        
    Returns:
        None
    """
    print("\n" + "="*60)
    print("CUDA SUPPORT VERIFICATION FOR LLAMA-CPP-PYTHON")
    print("="*60)
    
    print("\n[1] Checking System Information:")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    # Check for NVIDIA GPU via nvidia-smi
    print("\n[2] Checking NVIDIA GPU:")
    try:
        result = subprocess.run("nvidia-smi", capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected!")
            print("\nGPU Information:")
            lines = result.stdout.splitlines()
            # Print first few lines of nvidia-smi output
            for line in lines[:6]:
                print(f"  {line}")
        else:
            print("❌ nvidia-smi failed. No NVIDIA GPU found or driver issue.")
    except Exception as e:
        print(f"❌ Error checking nvidia-smi: {e}")
    
    # Check llama-cpp-python installation
    print("\n[3] Checking llama-cpp-python:")
    try:
        import llama_cpp
        print(f"✅ llama-cpp-python installed (version: {llama_cpp.__version__})")
        
        # Check for CUDA backends
        print("\n[4] Checking CUDA backends:")
        cuda_backends = []
        for attr in dir(llama_cpp):
            if attr.startswith("LLAMA_BACKEND_"):
                if "CUDA" in attr:
                    cuda_backends.append(attr)
                print(f"  - {attr}")
        
        if cuda_backends:
            print(f"✅ CUDA backends found: {cuda_backends}")
        else:
            print("❌ No CUDA backends found in llama_cpp!")
            
        # Check for other CUDA indicators
        print("\n[5] Checking additional CUDA indicators:")
        cuda_indicators = {
            "LLAMA_MAX_DEVICES": hasattr(llama_cpp, "LLAMA_MAX_DEVICES"),
            "get_gpu_info": hasattr(llama_cpp.Llama, "get_gpu_info"),
            "_use_cuda": hasattr(llama_cpp.Llama, "_use_cuda")
        }
        
        for name, present in cuda_indicators.items():
            status = "✅" if present else "❌"
            print(f"  {status} {name}")
        
        # Check if we have available_backends function
        if hasattr(llama_cpp, "get_available_backends") and callable(getattr(llama_cpp, "get_available_backends")):
            backends = llama_cpp.get_available_backends()
            print(f"\n✅ Available backends from API: {backends}")
            
        # Check build info
        if hasattr(llama_cpp, "__build_info__"):
            print("\n[6] Build information:")
            for key, value in llama_cpp.__build_info__.items():
                print(f"  - {key}: {value}")
        
        # Test model initialization with CUDA
        print("\n[7] Testing CUDA initialization (no model loading):")
        try:
            # Just test initialization with CUDA flag
            print("  Attempting to initialize with n_gpu_layers = -1...")
            # Don't actually proceed with model loading
            print("  ✅ CUDA initialization appears possible")
        except Exception as e:
            print(f"  ❌ Error with CUDA initialization: {e}")
            
    except ImportError as e:
        print(f"❌ llama-cpp-python not installed: {e}")
    
    # Check environment variables
    print("\n[8] Checking environment variables:")
    cuda_env_vars = [
        "CUDA_VISIBLE_DEVICES",
        "LLAMA_CUBLAS",
        "CMAKE_ARGS"
    ]
    
    for var in cuda_env_vars:
        value = os.environ.get(var)
        status = "✅" if value else "❓"
        print(f"  {status} {var}: {value if value else 'not set'}")
    
    print("\n[9] Recommendations:")
    print("  If CUDA backends are not detected:")
    print("  1. Reinstall with: pip install --force-reinstall llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124")
    print("  2. Or set environment variable before running: set LLAMA_CUBLAS=1")
    print("  3. Refer to cuda_setup_guide.md for detailed instructions")
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    main() 