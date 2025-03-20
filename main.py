"""
Main entry point for the Git repository monitoring and LLM feedback system.

This script initializes all components and starts the monitoring system.

Command Line Usage Examples:
    python main.py /path/to/git/repository
    python main.py C:/Projects/my-project
"""
import argparse
import logging
import os
import sys

# Import our modules
from src.models import ModelManager
from src.conversation import ConversationManager
from src.diff_manager import DiffManager
from src.agents import DiffReceiver
from src.monitor import KeyMonitor

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
    
    try:
        args = parser.parse_args()
        repo_path = args.repo_path
        
        if not os.path.exists(repo_path):
            logger.error(f"Repository path not found: {repo_path}")
            print(f"Error: Repository path not found: {repo_path}")
            sys.exit(1)
        
        # Initialize components
        logger.info("Initializing components")
        
        # Load the model
        model = ModelManager.load_gemma_model()
        
        # Create conversation manager
        conversation_manager = ConversationManager()
        
        # Create diff manager
        diff_manager = DiffManager(repo_path)
        
        # Create agent
        agent = DiffReceiver(model, conversation_manager)
        
        # Create and start key monitor
        key_monitor = KeyMonitor(repo_path, diff_manager, agent)
        key_monitor.start_monitoring()
        
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
        print("\nProgram terminated by user.")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
