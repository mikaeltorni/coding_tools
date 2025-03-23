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
import time
from typing import Dict, Any, Optional, Tuple

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
            "min_p": 0.01           # Works well according to documentation
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