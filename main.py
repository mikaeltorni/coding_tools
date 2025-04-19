"""
Main entry point for the Git repository monitoring and LLM feedback system.

This script initializes all components and starts the monitoring system.

Functions:
    main(): Main function that parses command line arguments and runs the program.

Command Line Usage Examples:
    python main.py /path/to/git/repository
    python main.py C:/Projects/my-project
    python main.py /path/to/git/repository1 /path/to/git/repository2
    python main.py /path/to/git/repository --server-url http://localhost:8080
    python main.py /path/to/git/repository --temperature 0.7 --top-p 0.95

Keyboard Controls:
    Press configured hotkey (default: alt+q) to get LLM feedback on current changes
    Press Ctrl+Space to automatically commit changes (with LLM-generated commit message)
    Press Ctrl+C to exit the program
"""
import argparse
import logging
import os
import sys
import time
from data.model_config import (
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_HOTKEY
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
        description='Monitor Git repositories and feed diffs to LLM on hotkey press.'
    )
    parser.add_argument('repo_paths', type=str, nargs='+', help='Paths to the Git repositories to monitor')
    parser.add_argument('--server-url', type=str, default='http://localhost:8080', 
                        help='URL of the llama server (default: http://localhost:8080)')
    parser.add_argument('--hotkey', type=str, default=DEFAULT_HOTKEY,
                        help=f'Hotkey combination to trigger LLM feedback (default: {DEFAULT_HOTKEY})')
    
    # Model configuration arguments - using defaults from model_config
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                        help=f'Temperature parameter for text generation (default: {DEFAULT_TEMPERATURE})')
    parser.add_argument('--max-tokens', type=int, default=DEFAULT_MAX_TOKENS,
                        help=f'Maximum number of tokens to generate (default: {DEFAULT_MAX_TOKENS})')
    parser.add_argument('--context-length', type=int, default=DEFAULT_CONTEXT_LENGTH,
                        help=f'Context length for the model (default: {DEFAULT_CONTEXT_LENGTH})')
    
    try:
        args = parser.parse_args()
        repo_paths = args.repo_paths
        server_url = args.server_url
        hotkey = args.hotkey
        
        # Validate each repository path
        valid_repos = []
        for repo_path in repo_paths:
            if not os.path.exists(repo_path):
                logger.warning(f"Repository path not found: {repo_path}")
                print(f"Warning: Repository path not found: {repo_path}")
                continue
            
            # Check if the path is a valid Git repository
            if not is_git_repo(repo_path):
                logger.warning(f"Not a valid Git repository: {repo_path}")
                print(f"Warning: Not a valid Git repository: {repo_path}")
                continue
                
            valid_repos.append(repo_path)
            
        if not valid_repos:
            logger.error("No valid Git repositories found.")
            print("Error: No valid Git repositories found.")
            sys.exit(1)
                
        logger.info(f"Monitoring {len(valid_repos)} valid repositories: {valid_repos}")
        print(f"Monitoring {len(valid_repos)} valid repositories:")
        for repo in valid_repos:
            print(f"  - {repo}")

        payload = {
            "generation_settings": {
                "temperature": args.temperature,
                "top_p": 0.9,
                "n_predict": args.max_tokens,
                "ctx_size": args.context_length
            }
        }
        
        # Initialize components
        logger.info("Initializing components")
        
        # Set up keyboard listener with multiple repositories
        setup_keyboard_listener(server_url, payload, valid_repos, hotkey)
        
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
