"""
test.py

A simple script for loading and interacting with the Gemma 3 1B model using llama-cpp-python.

Functions:
    setup_logging(): Configures the logging system
    check_cuda_support(): Checks if CUDA is available and enabled
    load_model(model_path): Loads the GGUF model using llama-cpp-python
    generate_response(model, prompt): Generates a response from the model
    main(): Main function to run the application

Command Line Usage Examples:
    python test.py
    python test.py --model_path "path/to/model.gguf"
"""
import os
import sys
import argparse
import logging
import time
import subprocess
import platform
from typing import Dict, Any, Optional, Tuple, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(funcName)s: %(message)s'
)

logger = logging.getLogger(__name__)

def setup_logging() -> None:
    """
    Configures the logging system for the application.
    
    Parameters:
        None
        
    Returns:
        None
    """
    logger.debug("Setting up logging")
    # Logging is already configured at the module level
    logger.info("Logging configured successfully")

def check_cuda_support() -> Tuple[bool, Dict[str, Any]]:
    """
    Checks if CUDA is available and enabled in llama-cpp-python.
    
    Parameters:
        None
        
    Returns:
        Tuple[bool, Dict[str, Any]]: Whether CUDA is supported and additional info
    """
    logger.info("Checking CUDA support...")
    
    cuda_info = {
        "available": False,
        "version": None,
        "devices": [],
        "llama_cpp_backends": []
    }
    
    # Check system CUDA availability using torch if available
    try:
        import torch
        cuda_info["available"] = torch.cuda.is_available()
        if cuda_info["available"]:
            cuda_info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            cuda_info["version"] = torch.version.cuda
            logger.info(f"CUDA available: {cuda_info['available']}, version: {cuda_info['version']}")
            logger.info(f"CUDA devices: {cuda_info['devices']}")
    except ImportError:
        logger.info("torch not available, using alternative CUDA detection")
    
    # Check if process can access NVIDIA GPU via nvidia-smi
    # This is the most reliable way to check if CUDA is actually available
    try:
        if platform.system() == "Windows":
            result = subprocess.run("nvidia-smi", capture_output=True, text=True, check=False)
            if result.returncode == 0:
                logger.info("NVIDIA GPU is accessible (nvidia-smi successful)")
                cuda_info["nvidia_smi_ok"] = True
                # If nvidia-smi works, we can force CUDA to be considered available
                cuda_info["available"] = True
            else:
                logger.warning("NVIDIA GPU not accessible or nvidia-smi not found")
                cuda_info["nvidia_smi_ok"] = False
    except Exception as e:
        logger.warning(f"Error checking nvidia-smi: {e}")
        cuda_info["nvidia_smi_ok"] = False
    
    # Check if llama-cpp-python has CUDA support
    try:
        import llama_cpp
        
        # Try different ways to check for CUDA support
        cuda_info["has_cuda_backend"] = hasattr(llama_cpp, "LLAMA_BACKEND_CUDA")
        
        # Check for other backends
        for attr in dir(llama_cpp):
            if attr.startswith("LLAMA_BACKEND_"):
                cuda_info["llama_cpp_backends"].append(attr)
        
        # Get package version
        cuda_info["llama_cpp_version"] = getattr(llama_cpp, "__version__", "unknown")
        
        # Log info about llama-cpp-python
        logger.info(f"llama-cpp-python version: {cuda_info['llama_cpp_version']}")
        logger.info(f"CUDA backend available in llama-cpp-python: {cuda_info['has_cuda_backend']}")
        logger.info(f"Available backends: {cuda_info['llama_cpp_backends']}")
        
        # Check if compiled with CUDA via requirements.txt
        try:
            with open('requirements.txt', 'r') as f:
                req_content = f.read()
                # Check for either cu124 or cu125 in the requirements
                if any(f'llama-cpp-python/whl/cu{ver}' in req_content for ver in ['124', '125']):
                    logger.info("llama-cpp-python was installed with CUDA support from pre-built wheel")
                    cuda_info["from_cuda_wheel"] = True
                    # If installed with CUDA wheel, override detection
                    cuda_info["available"] = True
        except Exception as e:
            logger.warning(f"Could not check requirements.txt: {e}")
        
        # Make the final decision on CUDA availability based on multiple factors
        cuda_available = (
            cuda_info.get("nvidia_smi_ok", False) or  # GPU is accessible
            cuda_info.get("has_cuda_backend", False) or  # Library has CUDA backend
            cuda_info.get("from_cuda_wheel", False) or  # Installed from CUDA wheel
            cuda_info.get("available", False)  # torch reported CUDA available
        )
                
        return cuda_available, cuda_info
        
    except ImportError as e:
        logger.error(f"Error importing llama_cpp: {e}")
        # If we can't import llama_cpp but nvidia-smi works, still consider CUDA available
        return cuda_info.get("nvidia_smi_ok", False), cuda_info

def load_model(model_path: str, force_cpu: bool = False) -> Any:
    """
    Loads the GGUF model using llama-cpp-python.
    
    Parameters:
        model_path (str): Path to the GGUF model file
        force_cpu (bool): Force CPU-only mode even if CUDA is available
        
    Returns:
        Any: The loaded model object
    """
    logger.debug(f"model_path: {model_path} | force_cpu: {force_cpu}")
    
    # First check CUDA support
    cuda_supported, cuda_info = check_cuda_support()
    
    try:
        from llama_cpp import Llama
    except ImportError as e:
        logger.error(f"Failed to import llama_cpp: {e}")
        logger.error("Make sure llama-cpp-python is installed with CUDA support")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)
    
    try:
        # Load model with CUDA support if available
        logger.info(f"Loading model from {model_path}")
        
        # GPU layers (0 = CPU only, -1 = all layers on GPU)
        n_gpu_layers = 0 if force_cpu else -1
        
        # OVERRIDE: Force GPU usage if nvidia-smi works, regardless of other detection
        if not force_cpu and cuda_info.get("nvidia_smi_ok", False):
            n_gpu_layers = -1
            cuda_supported = True
            logger.info("NVIDIA GPU detected via nvidia-smi, forcing GPU usage")
        
        # Print messaging based on detection
        if force_cpu:
            print("\n" + "="*60)
            print("WARNING: RUNNING IN CPU-ONLY MODE - EXPECT SLOW PERFORMANCE")
            print("Reason: Forced CPU mode by user")
            print(f"llama-cpp-python version: {cuda_info.get('llama_cpp_version', 'unknown')}")
            print("="*60 + "\n")
        elif cuda_supported:
            print("\n" + "="*60)
            print("CUDA SUPPORT DETECTED - Using GPU acceleration")
            print(f"llama-cpp-python version: {cuda_info.get('llama_cpp_version', 'unknown')}")
            if cuda_info.get("nvidia_smi_ok", False):
                print("NVIDIA GPU detected via nvidia-smi")
            print("="*60 + "\n")
        else:
            print("\n" + "="*60)
            print("WARNING: RUNNING IN CPU-ONLY MODE - EXPECT SLOW PERFORMANCE")
            print("Reason: CUDA not detected or not properly enabled")
            print(f"llama-cpp-python version: {cuda_info.get('llama_cpp_version', 'unknown')}")
            print("To enable CUDA, make sure:")
            print("1. You have a compatible NVIDIA GPU with proper drivers installed")
            print("2. llama-cpp-python is installed with CUDA support:")
            print("   pip install --force-reinstall llama-cpp-python --extra-index-url")
            print("   https://abetlen.github.io/llama-cpp-python/whl/cu124")
            print("="*60 + "\n")
        
        # Model parameters optimized for Gemma 3
        model_params = {
            "model_path": model_path,
            "n_ctx": 8192,                    # Context window size
            "n_batch": 1024,                  # Increased batch size for better throughput
            "n_gpu_layers": n_gpu_layers,     # GPU layers setting
            "seed": 42,                       # For reproducibility
            "verbose": True,                  # Enable verbose output to debug CUDA usage
            "f16_kv": True,                   # Use half precision for key/value cache
            "use_mlock": True,                # Lock memory to prevent swapping
        }
        
        # Load the model
        logger.info(f"Creating model with n_gpu_layers = {n_gpu_layers}")
        model = Llama(**model_params)
        
        # Log model loaded successfully
        logger.info("Model loaded successfully")
            
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)

def generate_response(model: Any, prompt: str) -> Tuple[Dict[str, Any], float, int]:
    """
    Generates a response from the model for the given prompt.
    
    Parameters:
        model (Any): The loaded model object
        prompt (str): The input prompt for the model
        
    Returns:
        Tuple[Dict[str, Any], float, int]: The model's response, tokens per second, and token count
    """
    logger.debug(f"prompt: {prompt}")
    
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    
    try:
        # Format the prompt according to Gemma 3's chat template
        # Note: llama.cpp adds <bos> automatically, so we don't need to add it
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        # Gemma 3 recommended inference settings
        generation_params = {
            "prompt": formatted_prompt,
            "max_tokens": 1024,
            "temperature": 1.0,     # Official recommended value
            "top_k": 64,            # Official recommended value
            "top_p": 0.95,          # Official recommended value
            "repeat_penalty": 1.0,  # Official recommended value
            "min_p": 0.01,          # Works well according to documentation
            "stream": False,        # Streaming disabled for token speed measurement
        }
        
        logger.info("Generating response...")
        
        # Start timing
        start_time = time.time()
        
        # Generate response
        response = model(**generation_params)
        
        # End timing
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Calculate tokens per second
        token_count = response.get("usage", {}).get("completion_tokens", 0)
        if "usage" not in response:
            # Try to get token count from different response formats
            if "choices" in response and len(response["choices"]) > 0:
                # Some versions of llama-cpp-python may return tokens differently
                choice = response["choices"][0]
                if "tokens_generated" in choice:
                    token_count = choice["tokens_generated"]
                # If we have the raw text, we can estimate tokens as words / 0.75
                elif "text" in choice and token_count == 0:
                    token_count = len(choice["text"].split()) // 0.75
        
        tokens_per_second = token_count / generation_time if generation_time > 0 else 0
        
        logger.info(f"Response generated successfully: {token_count} tokens in {generation_time:.2f} seconds ({tokens_per_second:.2f} tokens/sec)")
        return response, tokens_per_second, token_count
    
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise RuntimeError(f"Error generating response: {e}")

def main() -> None:
    """
    Main function to load the model and run an interactive chat session.
    
    Parameters:
        None
        
    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Run the Gemma 3 1B model with llama-cpp-python")
    parser.add_argument("--model_path", type=str, default="gemma-3-1b-it-Q4_K_M.gguf",
                        help="Path to the GGUF model file")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose logging")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU-only mode (disable CUDA)")
    parser.add_argument("--force-gpu", action="store_true", 
                        help="Force GPU usage regardless of CUDA detection")
    args = parser.parse_args()
    
    # Set log level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup_logging()
    
    try:
        # Load the model (override force_cpu if force-gpu is specified)
        force_cpu = args.cpu and not args.force_gpu
        model = load_model(args.model_path, force_cpu=force_cpu)
        
        print("\n===== Gemma 3 1B Interactive Chat =====")
        print("Type 'exit', 'quit', or press Ctrl+C to end the conversation.\n")
        
        # Interactive chat loop
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting chat. Goodbye!")
                break
            
            response, tokens_per_second, token_count = generate_response(model, user_input)
            
            # Extract the model's output text
            output_text = response["choices"][0]["text"] if "choices" in response else response["text"]
            print(f"\nGemma 3: {output_text}")
            
            # Display token generation statistics
            print(f"\n[Generation stats: {token_count} tokens, {tokens_per_second:.2f} tokens/sec]")
            
    except KeyboardInterrupt:
        print("\nExiting chat. Goodbye!")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 