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
import time
from data.model_config import (
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_HOTKEY,
    get_default_model_args
)
from src.keyboard_manager import (
    send_prompt_to_server,
    setup_keyboard_listener
)
from src.git_manager import (
    get_repo_diff,
    is_git_repo
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s:%(funcName)s: %(message)s'
)
logger = logging.getLogger(__name__)

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
    
    # Model configuration arguments - using defaults from model_config
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
        
        # Check if the path is a valid Git repository
        if not is_git_repo(repo_path):
            logger.error(f"Not a valid Git repository: {repo_path}")
            print(f"Error: Not a valid Git repository: {repo_path}")
            sys.exit(1)
        
        # Create model args dictionary
        model_args = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "repeat_penalty": args.repeat_penalty,
            "system_prompt": open(os.path.join("data", "prompts", "system", "diff_analyzer.xml")).read(),
            "repo_path": repo_path  # Add repo_path to model_args for the hotkey handler
        }
        
        # Initialize components
        logger.info("Initializing components")
        
        # Get diff from repository
        logger.info(f"Getting diff from repository: {repo_path}")
        diff_content = get_repo_diff(repo_path)
        
        # Test server connectivity with the diff content
        logger.info(f"Testing connection to LLM server at {server_url}")
        test_prompt = f"Testing server connectivity. Here is the current diff from the repository:\n\n{diff_content}"
        test_response = send_prompt_to_server(server_url, test_prompt, model_args)
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
