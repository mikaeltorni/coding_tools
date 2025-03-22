"""
Models module for loading and configuring language models.

This module handles the loading and configuration of different language models.

Functions:
    load_gemma_model(): Loads the Gemma LLM model
    load_quantized_gemma_model(): Loads the Gemma LLM model with 4-bit quantization
"""
import torch
import time
from transformers import pipeline, BitsAndBytesConfig, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages the loading and configuration of language models.
    """
    
    @staticmethod
    def load_gemma_model(config=None):
        """
        Loads and initializes the Gemma LLM model.
        
        Parameters:
            config (dict, optional): Configuration parameters for the model
            
        Returns:
            tuple: (pipeline, config) - Initialized pipeline and its configuration
        """
        logger.debug("Loading Gemma model")
        
        # Default configuration
        default_config = {
            "model_name": "google/gemma-3-1b-it",
            "revision": "b13e02e0952a32651f3445bc26517c999a1a928b",
            "device": "cuda",
            "torch_dtype": torch.bfloat16,
            "max_new_tokens": 16384,
            "do_sample": True,
            "top_p": 0.9,            
            "temperature": 0.01
        }
        
        # Use provided config or default
        config = config or default_config
        
        try:
            # Prepare pipeline arguments
            pipeline_args = {
                "task": "text-generation",
                "model": config["model_name"],
                "device": config["device"],
                "torch_dtype": config["torch_dtype"]
            }
            
            # Add revision parameter if specified
            if config.get("revision"):
                logger.debug(f"Using specific model revision: {config['revision']}")
                pipeline_args["revision"] = config["revision"]
            
            # Create the pipeline with the arguments
            pipe = pipeline(**pipeline_args)
            
            logger.debug("Model loaded successfully")
            return pipe, config
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model: {e}") 

    @staticmethod
    def load_quantized_gemma_model(config=None, quantization_type="nf4", use_double_quant=False):
        """
        Loads and initializes the Gemma LLM model with 4-bit quantization.
        
        Parameters:
            config (dict, optional): Configuration parameters for the model
            quantization_type (str, optional): Type of 4-bit quantization, either "nf4" or "fp4"
            use_double_quant (bool, optional): Whether to use nested quantization for additional memory savings
            
        Returns:
            tuple: (model, config) - Initialized model and its configuration
        """
        logger.debug(f"Loading quantized Gemma model with {quantization_type} and double_quant={use_double_quant}")
        
        # Default configuration
        default_config = {
            "model_name": "google/gemma-3-1b-it",
            "revision": "b13e02e0952a32651f3445bc26517c999a1a928b",
            "device_map": "auto",  # Let transformers determine optimal device mapping
            "max_new_tokens": 16384,
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.01
        }
        
        # Use provided config or default
        config = config or default_config
        
        try:
            # Validate quantization type
            if quantization_type not in ["nf4", "fp4"]:
                raise ValueError(f"Unsupported quantization type: {quantization_type}. Use 'nf4' or 'fp4'")
                
            # Create quantization configuration
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quantization_type,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=use_double_quant
            )
            
            logger.debug("Creating quantized model with BitsAndBytesConfig")
            
            # Load the model with quantization config
            model_args = {
                "pretrained_model_name_or_path": config["model_name"],
                "quantization_config": quantization_config,
                "device_map": config["device_map"],
                "torch_dtype": "auto"  # Let the model decide the best dtype based on the quantization
            }
            
            # Add revision parameter if specified
            if config.get("revision"):
                logger.debug(f"Using specific model revision: {config['revision']}")
                model_args["revision"] = config["revision"]
            
            # Load the model
            model = AutoModelForCausalLM.from_pretrained(**model_args)
            
            logger.debug("Quantized model loaded successfully")
            return model, config
        except Exception as e:
            logger.error(f"Error loading quantized model: {e}")
            raise RuntimeError(f"Failed to load quantized model: {e}") 