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

# Import our modules
# from src.conversation import ConversationManager
# from src.diff_manager import DiffManager
# from src.agents import DiffReceiver
# from src.monitor import KeyMonitor

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
        
        if not os.path.exists(repo_path):
            logger.error(f"Repository path not found: {repo_path}")
            print(f"Error: Repository path not found: {repo_path}")
            sys.exit(1)
        
        # Initialize components
        logger.info("Initializing components")
        
        # # Connect to the llama server
        # logger.info(f"Connecting to llama server at: {server_url}")
        # model_tuple = ModelManager.create_server_client(server_url, model_config)
        
        # # Create conversation manager
        # conversation_manager = ConversationManager()
        
        # # Create diff manager
        # diff_manager = DiffManager(repo_path)
        
        # # Create agent
        # agent = DiffReceiver(model_tuple, conversation_manager)
        
        # # Create and start key monitor
        # key_monitor = KeyMonitor(repo_path, diff_manager, agent)
        # key_monitor.start_monitoring()

        payload_with_parameters = {
            "prompt": "Testing the llama server",
            "n_predict": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repeat_penalty": args.repeat_penalty
        }
        response = requests.post(f"{server_url}/completion", json=payload_with_parameters)
        print(response.json()["content"])
        
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
        print("\nProgram terminated by user.")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
