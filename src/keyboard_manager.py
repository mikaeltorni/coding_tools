"""
keyboard_manager.py

Manages keyboard hotkey detection and handling for LLM feedback.

Functions:
    send_prompt_to_server(server_url, prompt, model_args): Sends a prompt to the LLM server
    handle_hotkey_press(server_url, model_args): Handles the hotkey press event
    setup_keyboard_listener(server_url, model_args, hotkey): Sets up keyboard listener
"""
import logging
import requests
import keyboard
import os
from data.model_config import (
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_HOTKEY
)
from src.git_manager import get_repo_diff

# Configure logging
logger = logging.getLogger(__name__)

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
            "repeat_penalty": model_args.get("repeat_penalty", DEFAULT_REPEAT_PENALTY),
            "system_prompt": model_args.get("system_prompt", "")
        }
        
        response = requests.post(f"{server_url}/completion", json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        result = response.json()
        logger.debug(f"response received | content length: {len(result.get('content', ''))}")
        return result.get("content", "")
    
    except requests.RequestException as e:
        logger.error(f"Error sending request to server: {e}")
        raise RuntimeError(f"Failed to communicate with LLM server: {e}")

def save_diff_to_file(diff_content, output_file="output.txt"):
    """
    Save diff content to a file.
    
    Parameters:
        diff_content (str): Diff content to save
        output_file (str): Path to the output file, default is 'output.txt'
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.debug(f"Saving diff content to file: {output_file}")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(diff_content)
        logger.info(f"Successfully saved diff content to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving diff content to file: {e}")
        return False

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
    
    try:
        # Get repository path from model_args
        repo_path = model_args.get("repo_path", "")
        
        if not repo_path:
            logger.warning("Repository path not found in model_args")
            prompt = "User pressed the hotkey. Please provide feedback on the current code."
        else:
            # Get diff from the repository
            logger.info(f"Getting diff from repository: {repo_path}")
            diff_content = get_repo_diff(repo_path)
            
            # Save diff content to output.txt
            if diff_content and diff_content != "No changes detected in the repository.":
                save_diff_to_file(diff_content)
                print(f"Diff content saved to output.txt")
            
            # Create a prompt with the diff content
            if diff_content:
                prompt = f"{diff_content}"
            else:
                prompt = "User pressed the hotkey, but no changes were found in the repository."
        
        # Send the prompt to the server
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