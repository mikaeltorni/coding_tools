"""
test.py

A simple script for loading and interacting with the Gemma 3 1B model using llama-cpp-python.

Functions:
    setup_logging(): Configures the logging system
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
from typing import Dict, Any, Optional

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

def load_model(model_path: str) -> Any:
    """
    Loads the GGUF model using llama-cpp-python.
    
    Parameters:
        model_path (str): Path to the GGUF model file
        
    Returns:
        Any: The loaded model object
    """
    logger.debug(f"model_path: {model_path}")
    
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
        
        # Model parameters optimized for Gemma 3
        model_params = {
            "model_path": model_path,
            "n_ctx": 8192,          # Context window size
            "n_batch": 512,         # Batch size for prompt processing
            "n_gpu_layers": -1,     # -1 means offload all layers to GPU if possible
            "seed": 42,             # For reproducibility
            "verbose": False        # Set to True for more debug information
        }
        
        model = Llama(**model_params)
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)

def generate_response(model: Any, prompt: str) -> Dict[str, Any]:
    """
    Generates a response from the model for the given prompt.
    
    Parameters:
        model (Any): The loaded model object
        prompt (str): The input prompt for the model
        
    Returns:
        Dict[str, Any]: The model's response
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
            "min_p": 0.01           # Works well according to documentation
        }
        
        logger.info("Generating response...")
        response = model(**generation_params)
        logger.info("Response generated successfully")
        return response
    
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
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        # Load the model
        model = load_model(args.model_path)
        
        print("\n===== Gemma 3 1B Interactive Chat =====")
        print("Type 'exit', 'quit', or press Ctrl+C to end the conversation.\n")
        
        # Interactive chat loop
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting chat. Goodbye!")
                break
            
            response = generate_response(model, user_input)
            
            # Extract the model's output text
            output_text = response["choices"][0]["text"] if "choices" in response else response["text"]
            print(f"\nGemma 3: {output_text}")
            
    except KeyboardInterrupt:
        print("\nExiting chat. Goodbye!")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 