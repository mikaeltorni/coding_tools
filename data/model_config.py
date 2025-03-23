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
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40
DEFAULT_REPEAT_PENALTY = 0
DEFAULT_CONTEXT_LENGTH = 32768
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
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "top_p": DEFAULT_TOP_P,
        "top_k": DEFAULT_TOP_K,
        "repeat_penalty": DEFAULT_REPEAT_PENALTY,
        "context_length": DEFAULT_CONTEXT_LENGTH,
        "frequency_penalty": DEFAULT_FREQUENCY_PENALTY,
        "presence_penalty": DEFAULT_PRESENCE_PENALTY
    }
    
    logger.debug(f"Default model arguments: {default_args}")
    return default_args 