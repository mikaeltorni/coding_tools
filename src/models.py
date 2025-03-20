"""
Models module for loading and configuring language models.

This module handles the loading and configuration of different language models.

Functions:
    load_gemma_model(): Loads the Gemma LLM model
"""
import torch
from transformers import pipeline
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
            "device": "cuda",
            "torch_dtype": torch.bfloat16,
            "max_new_tokens": 16384,
            "temperature": 0.01
        }
        
        # Use provided config or default
        config = config or default_config
        
        try:
            pipe = pipeline(
                "text-generation",
                model=config["model_name"],
                device=config["device"],
                torch_dtype=config["torch_dtype"]
            )
            logger.debug("Model loaded successfully")
            return pipe, config
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model: {e}") 