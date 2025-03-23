"""
model_config.py

Contains configuration parameters for LLM model interactions.

Functions:
    get_default_model_args(): Returns a dictionary with default model parameters
"""
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Default model parameters
DEFAULT_TEMP = 0
DEFAULT_N_PREDICT = 4096
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40
DEFAULT_REPEAT_PENALTY = 1.0
DEFAULT_CTX_SIZE = 4096
DEFAULT_FREQUENCY_PENALTY = 0
DEFAULT_PRESENCE_PENALTY = 0

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
        "temp": DEFAULT_TEMP,
        "n_predict": DEFAULT_N_PREDICT,
        "top_p": DEFAULT_TOP_P,
        "top_k": DEFAULT_TOP_K,
        "repeat_penalty": DEFAULT_REPEAT_PENALTY,
        "ctx_size": DEFAULT_CTX_SIZE,
        "frequency_penalty": DEFAULT_FREQUENCY_PENALTY,
        "presence_penalty": DEFAULT_PRESENCE_PENALTY
    }
    
    logger.debug(f"Default model arguments: {default_args}")
    return default_args 