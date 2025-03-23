"""
model_config.py

Contains configuration parameters for LLM model interactions.

Functions:
    None.
"""
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Default model parameters
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 200
DEFAULT_CONTEXT_LENGTH = 32768

# Keyboard configuration
DEFAULT_HOTKEY = "ctrl+space"
