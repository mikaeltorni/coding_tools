"""
Main script for monitoring Git repository changes and feeding diffs to the Gemma LLM.

This script monitors a Git repository and sends diffs to a Gemma LLM when CTRL+5 is pressed.
It maintains context between interactions and prepares for multi-agent functionality.

Functions:
    load_model(): Loads the LLM model
    get_git_diffs(repo_path): Gets current diffs from the Git repository
    send_message(pipe, messages): Sends messages to the LLM and returns response
    update_conversation_context(context, role, content): Updates the conversation history
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

# Global variables
conversation_context = []
file_contexts = {}  # Stores context for each edited file

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
        dict: Dictionary mapping filenames to their diff content
    """
    logger.debug(f"Getting diffs from: {repo_path}")
    
    try:
        repo = Repo(repo_path)
        
        # Check both staged and unstaged changes
        has_changes = repo.is_dirty(untracked_files=True)
        
        if not has_changes and not repo.untracked_files:
            logger.info("No changes detected in repository")
            return "No changes detected in repository.", {}
        
        # Get diffs for all modified files
        diffs = []
        file_diffs = {}  # Dictionary to store diffs by filename
        
        # Get unstaged changes (working tree changes)
        for diff_item in repo.index.diff(None):
            try:
                file_diff = repo.git.diff(diff_item.a_path)
                diffs.append(f"File: {diff_item.a_path}\n{file_diff}\n")
                file_diffs[diff_item.a_path] = file_diff
            except Exception as e:
                logger.error(f"Error getting diff for {diff_item.a_path}: {e}")
        
        # Get staged changes (index changes)
        for diff_item in repo.index.diff('HEAD'):
            try:
                file_diff = repo.git.diff('--cached', diff_item.a_path)
                diffs.append(f"File (staged): {diff_item.a_path}\n{file_diff}\n")
                file_diffs[f"{diff_item.a_path} (staged)"] = file_diff
            except Exception as e:
                logger.error(f"Error getting staged diff for {diff_item.a_path}: {e}")
        
        # Add untracked files
        untracked = repo.untracked_files
        if untracked:
            untracked_text = "Untracked files:\n" + "\n".join(untracked)
            diffs.append(untracked_text)
            file_diffs["untracked_files"] = untracked_text
        
        if not diffs:
            logger.warning("Repository is marked as dirty but no diffs were found")
            return "Repository has changes, but no specific diffs were detected.", {}
            
        return "\n".join(diffs), file_diffs
    except Exception as e:
        logger.error(f"Error accessing Git repository: {e}")
        return f"Error accessing Git repository: {e}", {}

def send_message(pipe, messages):
    """
    Sends messages to the LLM and returns the response.
    
    Parameters:
        pipe (pipeline): Initialized HuggingFace pipeline
        messages (list): List of message dictionaries to send
        
    Returns:
        str: Response from the LLM
    """
    logger.debug(f"Sending messages to model: {len(messages)} messages")
    try:
        output = pipe(text_inputs=messages, max_new_tokens=16384, temperature=0.01)
        response = output[0]["generated_text"][-1]["content"]
        logger.debug("Received response from model")
        return response
    except Exception as e:
        logger.error(f"Error sending message to model: {e}")
        return f"Error: Could not get a response from the model: {e}"

def update_conversation_context(messages, role, content):
    """
    Updates the conversation context with a new message.
    
    Parameters:
        messages (list): The current conversation context
        role (str): The role of the message (system, user, assistant)
        content (str): The content of the message
        
    Returns:
        list: Updated conversation context
    """
    messages.append({
        "role": role,
        "content": [{"type": "text", "text": content}]
    })
    return messages

def update_file_contexts(file_diffs):
    """
    Updates the context for each file that has been edited.
    
    Parameters:
        file_diffs (dict): Dictionary mapping filenames to their diff content
        
    Returns:
        None
    """
    global file_contexts
    logger.debug(f"Updating context for {len(file_diffs)} files")
    
    for filename, diff in file_diffs.items():
        if filename in file_contexts:
            # Update existing context
            file_contexts[filename]["history"].append(diff)
            # Keep only the last 5 diffs to prevent context from growing too large
            file_contexts[filename]["history"] = file_contexts[filename]["history"][-5:]
        else:
            # Create new context
            file_contexts[filename] = {
                "history": [diff],
                "last_updated": 0
            }
    
    logger.debug(f"File contexts updated. Total tracked files: {len(file_contexts)}")

def get_system_prompt():
    """
    Returns the system prompt for the DiffReceiver agent.
    
    Parameters:
        None
        
    Returns:
        str: System prompt
    """
    return """You are a helpful DiffReceiver agent that analyzes Git diffs.
Your role is to understand code changes and provide useful feedback or explanations.
Be concise and focus on the most important aspects of the changes.
Provide constructive feedback and suggestions for improvement when appropriate."""

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
    global conversation_context
    
    # Initialize conversation context with system prompt
    if not conversation_context:
        conversation_context = [{
            "role": "system",
            "content": [{"type": "text", "text": get_system_prompt()}]
        }]
    
    def on_hotkey():
        logger.info("Hotkey detected, processing diffs")
        
        try:
            # Get the current diffs
            diffs, file_diffs = get_git_diffs(repo_path)
            
            if diffs and diffs != "No changes detected in repository.":
                # Update file contexts
                update_file_contexts(file_diffs)
                
                # Add user message to conversation context
                update_conversation_context(
                    conversation_context, 
                    "user", 
                    f"Here are the current changes in my Git repository:\n\n{diffs}"
                )
                
                # Send message and get response
                response = send_message(pipe, conversation_context)
                
                # Add assistant response to conversation context
                update_conversation_context(conversation_context, "assistant", response)
                
                # Display response
                print("\n\n========== MODEL RESPONSE ==========\n")
                print(response)
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
