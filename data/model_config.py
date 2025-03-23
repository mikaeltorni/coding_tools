"""
model_config.py

Contains configuration parameters for LLM model interactions.

Functions:
    get_default_model_args(): Returns a dictionary with default model parameters
"""
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Model type configuration
DEFAULT_MODEL_TYPE = "llama_server"

# Default model parameters
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40
DEFAULT_REPEAT_PENALTY = 0

# Keyboard configuration
DEFAULT_HOTKEY = "ctrl+space"

def get_default_model_args():
    """
    Returns a dictionary containing default model parameters.
    
    Parameters:
        None
        
    Returns:
        dict: Dictionary with default model parameters
    """
    logger.debug("Retrieving default model arguments")
    
    default_args = {
        "model_type": DEFAULT_MODEL_TYPE,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "top_p": DEFAULT_TOP_P,
        "top_k": DEFAULT_TOP_K,
        "repeat_penalty": DEFAULT_REPEAT_PENALTY
    }
    
    logger.debug(f"Default model arguments: {default_args}")
    return default_args 