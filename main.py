"""
Main entry point for the Git repository monitoring and LLM feedback system.

This script initializes all components and starts the monitoring system.

Command Line Usage Examples:
    python main.py /path/to/git/repository
    python main.py C:/Projects/my-project
    python main.py /path/to/git/repository --server-url http://localhost:8080
    python main.py /path/to/git/repository --temperature 0.7 --top-p 0.95
"""
import argparse
import logging
import os
import sys
import requests
import keyboard
import time

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s:%(funcName)s: %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_TYPE = "llama_server"
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40
DEFAULT_REPEAT_PENALTY = 0
DEFAULT_HOTKEY = "ctrl+space"

def send_prompt_to_server(server_url, prompt, model_args):
    """
    Send a prompt to the LLM server and return the response.
    
    Parameters:
        server_url (str): URL of the LLM server
        prompt (str): Text prompt to send to the LLM
        model_args (dict): Model configuration parameters
        
    Returns:
        str: LLM response content
    """
    logger.debug(f"server_url: {server_url} | prompt length: {len(prompt)}")
    
    try:
        payload = {
            "prompt": prompt,
            "n_predict": model_args.get("max_tokens", DEFAULT_MAX_TOKENS),
            "temperature": model_args.get("temperature", DEFAULT_TEMPERATURE),
            "top_p": model_args.get("top_p", DEFAULT_TOP_P),
            "top_k": model_args.get("top_k", DEFAULT_TOP_K),
            "repeat_penalty": model_args.get("repeat_penalty", DEFAULT_REPEAT_PENALTY)
        }
        
        response = requests.post(f"{server_url}/completion", json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        result = response.json()
        logger.debug(f"response received | content length: {len(result.get('content', ''))}")
        return result.get("content", "")
    
    except requests.RequestException as e:
        logger.error(f"Error sending request to server: {e}")
        raise RuntimeError(f"Failed to communicate with LLM server: {e}")

def handle_hotkey_press(server_url, model_args):
    """
    Callback function for hotkey press event.
    
    Parameters:
        server_url (str): URL of the LLM server
        model_args (dict): Model configuration parameters
        
    Returns:
        None
    """
    logger.info("Hotkey pressed - sending prompt to LLM")
    prompt = "User pressed the hotkey. Please provide feedback on the current code."
    
    try:
        response = send_prompt_to_server(server_url, prompt, model_args)
        print("\n--- LLM Response ---")
        print(response)
        print("-------------------\n")
    except Exception as e:
        logger.error(f"Error handling hotkey press: {e}")
        print(f"Error: {e}")

def setup_keyboard_listener(server_url, model_args, hotkey=DEFAULT_HOTKEY):
    """
    Set up keyboard listener for the specified hotkey.
    
    Parameters:
        server_url (str): URL of the LLM server
        model_args (dict): Model configuration parameters
        hotkey (str): Keyboard hotkey combination to trigger LLM prompt
        
    Returns:
        None
    """
    logger.debug(f"Setting up keyboard listener for hotkey: {hotkey}")
    
    try:
        # Create a callback that includes the server_url and model_args
        def hotkey_callback():
            handle_hotkey_press(server_url, model_args)
        
        # Register the hotkey
        keyboard.add_hotkey(hotkey, hotkey_callback)
        logger.info(f"Keyboard listener set up for hotkey: {hotkey}")
        print(f"Press {hotkey} to get LLM feedback (Ctrl+C to exit)")
    
    except Exception as e:
        logger.error(f"Error setting up keyboard listener: {e}")
        raise RuntimeError(f"Failed to set up keyboard listener: {e}")

def main():
    """
    Main function that parses command line arguments and runs the program.
    
    Parameters:
        None
        
    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description='Monitor Git repository and feed diffs to LLM on hotkey press.'
    )
    parser.add_argument('repo_path', type=str, help='Path to the Git repository to monitor')
    parser.add_argument('--server-url', type=str, default='http://localhost:8080', 
                        help='URL of the llama server (default: http://localhost:8080)')
    parser.add_argument('--hotkey', type=str, default=DEFAULT_HOTKEY,
                        help=f'Hotkey combination to trigger LLM feedback (default: {DEFAULT_HOTKEY})')
    
    # Model configuration arguments - using defaults from ModelConfig class
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                        help=f'Temperature parameter for text generation (default: {DEFAULT_TEMPERATURE})')
    parser.add_argument('--top-p', type=float, default=DEFAULT_TOP_P,
                        help=f'Top-p sampling parameter (default: {DEFAULT_TOP_P})')
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K,
                        help=f'Top-k sampling parameter (default: {DEFAULT_TOP_K})')
    parser.add_argument('--max-tokens', type=int, default=DEFAULT_MAX_TOKENS,
                        help=f'Maximum number of tokens to generate (default: {DEFAULT_MAX_TOKENS})')
    parser.add_argument('--repeat-penalty', type=float, default=DEFAULT_REPEAT_PENALTY,
                        help=f'Penalty for repeated tokens (default: {DEFAULT_REPEAT_PENALTY})')
    
    try:
        args = parser.parse_args()
        repo_path = args.repo_path
        server_url = args.server_url
        hotkey = args.hotkey
        
        if not os.path.exists(repo_path):
            logger.error(f"Repository path not found: {repo_path}")
            print(f"Error: Repository path not found: {repo_path}")
            sys.exit(1)
        
        # Create model args dictionary
        model_args = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "repeat_penalty": args.repeat_penalty
        }
        
        # Initialize components
        logger.info("Initializing components")
        
        # Test server connectivity
        logger.info(f"Testing connection to LLM server at {server_url}")
        test_response = send_prompt_to_server(server_url, "Testing server connectivity", model_args)
        logger.info("Successfully connected to LLM server")
        
        # Set up keyboard listener
        setup_keyboard_listener(server_url, model_args, hotkey)
        
        # Keep the program running until Ctrl+C is pressed
        print("Monitoring keyboard. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(0.1)  # Reduce CPU usage
        except KeyboardInterrupt:
            logger.info("Program terminated by user")
            print("\nProgram terminated by user.")
        
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
        print("\nProgram terminated by user.")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
