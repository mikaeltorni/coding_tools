"""
Models module for loading and configuring language models.

This module handles the loading and configuration of language models using llama.cpp.

Functions:
    load_model(): Loads a model using llama.cpp for optimized inference
"""
import torch
import time
import logging
from pathlib import Path
import os

# Import llama-cpp-python
from llama_cpp import Llama

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages the loading and configuration of language models.
    """
    
    @staticmethod
    def load_model(model_path=None, config=None):
        """
        Loads and initializes a model using llama.cpp for optimized inference.
        
        Parameters:
            model_path (str): Path to the GGUF model file
            config (dict, optional): Configuration parameters for the model
            
        Returns:
            tuple: (model, config) - Initialized llama.cpp model and its configuration
        """
        logger.debug("Loading model with llama.cpp")
        
        # Default configuration with optimized settings
        default_config = {
            "n_gpu_layers": -1,  # -1 means all layers on GPU
            "n_ctx": 8192,       # Context window size
            "n_batch": 512,      # Batch size for prompt processing
            "n_threads": 8,      # Number of CPU threads to use
            "f16_kv": True,      # Use half-precision for KV cache
            "use_mlock": True,   # Lock memory to prevent swapping to disk
            "max_tokens": 16384, # Maximum number of tokens to generate
            "top_p": 0.9,        # Top-p sampling
            "top_k": 40,         # Top-k sampling
            "temperature": 0.01, # Temperature for sampling
            "repeat_penalty": 1.1, # Penalty for repeated tokens
            "verbose": False     # Verbose output
        }
        
        # Use provided config or default, with provided values overriding defaults
        merged_config = default_config.copy()
        if config:
            merged_config.update(config)
        
        # Check if model path is provided
        if not model_path:
            raise ValueError("Model path must be provided for llama.cpp model loading")
        
        # Validate model path
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Determine the number of available CPU cores if not specified
            if merged_config.get("n_threads") is None:
                merged_config["n_threads"] = os.cpu_count() or 4
                logger.debug(f"Using {merged_config['n_threads']} CPU threads")
            
            # Check for CUDA device if not specified
            if merged_config.get("n_gpu_layers") is None:
                if torch.cuda.is_available():
                    merged_config["n_gpu_layers"] = -1  # Use all layers on GPU
                    logger.debug("CUDA available, offloading all layers to GPU")
                else:
                    merged_config["n_gpu_layers"] = 0  # CPU only
                    logger.debug("CUDA not available, using CPU only")
            
            # Load the model with llama.cpp
            logger.debug(f"Creating llama.cpp model with config: {merged_config}")
            
            # Create the model arguments by extracting only the parameters that llama.cpp accepts
            llama_args = {
                "model_path": str(model_path),
                "n_gpu_layers": merged_config["n_gpu_layers"],
                "n_ctx": merged_config["n_ctx"],
                "n_batch": merged_config["n_batch"],
                "n_threads": merged_config["n_threads"],
                "f16_kv": merged_config["f16_kv"],
                "use_mlock": merged_config["use_mlock"],
                "verbose": merged_config["verbose"]
            }
            
            # Create the model object
            model = Llama(**llama_args)
            
            logger.debug("llama.cpp model loaded successfully")
            # Explicitly return a tuple of (model, config)
            return (model, merged_config)
        except Exception as e:
            logger.error(f"Error loading llama.cpp model: {e}")
            raise RuntimeError(f"Failed to load llama.cpp model: {e}") 