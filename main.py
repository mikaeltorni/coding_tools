"""
Main script for monitoring Git repository changes and feeding diffs to the Gemma LLM.

This script monitors a Git repository and sends diffs to a Gemma LLM when CTRL+5 is pressed.

Functions:
    load_model(): Loads the LLM model
    get_git_diffs(repo_path): Gets current diffs from the Git repository
    monitor_hotkey(repo_path): Monitors for hotkey press and processes diffs
    main(): Main function that parses arguments and runs the program

Command Line Usage Examples:
    python main.py /path/to/git/repository
    python main.py C:/Projects/my-project
"""
import torch
import argparse
import threading
import sys
import os
from git import Repo
import keyboard
from transformers import pipeline

import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s:%(funcName)s: %(message)s'
)
logger = logging.getLogger(__name__)

def load_model():
    """
    Loads and initializes the LLM model.
    
    Parameters:
        None
        
    Returns:
        pipeline: Initialized HuggingFace pipeline for text generation
    """
    logger.debug("Loading model")
    try:
        pipe = pipeline(
            "text-generation",
            model="google/gemma-3-1b-it",
            device="cuda",
            torch_dtype=torch.bfloat16
        )
        logger.debug("Model loaded successfully")
        return pipe
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

def get_git_diffs(repo_path):
    """
    Gets the current uncommitted diffs from a Git repository.
    
    Parameters:
        repo_path (str): Path to the Git repository
        
    Returns:
        str: Text representation of all current diffs
    """
    logger.debug(f"Getting diffs from: {repo_path}")
    
    try:
        repo = Repo(repo_path)
        
        if not repo.is_dirty():
            logger.info("No changes detected in repository")
            return "No changes detected in repository."
        
        # Get diffs for all modified files
        diffs = []
        for diff_item in repo.index.diff(None):
            try:
                file_diff = repo.git.diff(diff_item.a_path)
                diffs.append(f"File: {diff_item.a_path}\n{file_diff}\n")
            except Exception as e:
                logger.error(f"Error getting diff for {diff_item.a_path}: {e}")
        
        # Add untracked files
        untracked = repo.untracked_files
        if untracked:
            diffs.append("Untracked files:\n" + "\n".join(untracked))
        
        if not diffs:
            return "No meaningful changes detected."
            
        return "\n".join(diffs)
    except Exception as e:
        logger.error(f"Error accessing Git repository: {e}")
        return f"Error accessing Git repository: {e}"

def monitor_hotkey(repo_path, pipe):
    """
    Monitors for the CTRL+5 hotkey and processes Git diffs when pressed.
    
    Parameters:
        repo_path (str): Path to the Git repository
        pipe (pipeline): Initialized HuggingFace pipeline
        
    Returns:
        None
    """
    logger.debug(f"Starting hotkey monitor for repository: {repo_path}")
    
    def on_hotkey():
        logger.info("Hotkey detected, processing diffs")
        
        try:
            # Get the current diffs
            diffs = get_git_diffs(repo_path)
            
            if diffs:
                # Prepare messages for the model
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."},]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Here are the current changes in my Git repository:\n\n{diffs}"}
                        ]
                    }
                ]
                
                # Generate response
                logger.debug("Sending diffs to model")
                output = pipe(text_inputs=messages, max_new_tokens=16384, temperature=0.01)
                print("\n\n========== MODEL RESPONSE ==========\n")
                print(output[0]["generated_text"][-1]["content"])
                print("\n========== END RESPONSE ==========\n")
            else:
                print("No changes to process.")
        except Exception as e:
            logger.error(f"Error processing diffs: {e}")
    
    # Register the hotkey
    try:
        keyboard.add_hotkey('ctrl+5', on_hotkey)
        logger.info("Hotkey registered: CTRL+5")
        print("Monitoring Git repository. Press CTRL+5 to get AI feedback on current changes.")
        print("Press CTRL+C to exit.")
        
        # Keep the program running
        keyboard.wait('ctrl+c')
    except Exception as e:
        logger.error(f"Error setting up hotkey monitoring: {e}")
        raise RuntimeError(f"Failed to set up hotkey monitoring: {e}")

def main():
    """
    Main function that parses command line arguments and runs the program.
    
    Parameters:
        None
        
    Returns:
        None
    """
    parser = argparse.ArgumentParser(description='Monitor Git repository and feed diffs to LLM on hotkey press.')
    parser.add_argument('repo_path', type=str, help='Path to the Git repository to monitor')
    
    try:
        args = parser.parse_args()
        repo_path = args.repo_path
        
        if not os.path.exists(repo_path):
            logger.error(f"Repository path not found: {repo_path}")
            print(f"Error: Repository path not found: {repo_path}")
            sys.exit(1)
            
        # Load the model
        pipe = load_model()
        
        # Start monitoring
        monitor_hotkey(repo_path, pipe)
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
        print("\nProgram terminated by user.")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
