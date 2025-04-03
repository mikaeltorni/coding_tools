"""
keyboard_manager.py

Manages keyboard hotkey detection and handling for LLM feedback.

Functions:
    send_prompt_to_server(server_url, prompt, model_args): Sends a prompt to the LLM server
    handle_hotkey_press(server_url, model_args): Handles the hotkey press event
    handle_commit_hotkey(repo_path): Handles the commit hotkey (Ctrl+Space) press event
    setup_keyboard_listener(server_url, model_args, hotkey): Sets up keyboard listener
"""
import logging
import keyboard
import openai
import os
import json
from pathlib import Path

from data.model_config import (
    DEFAULT_HOTKEY
)
from src.git_manager import get_repo_diff, commit_changes

# Configure logging
logger = logging.getLogger(__name__)

def send_prompt_to_server(server_url, payload, repo_path):
    """
    Send a prompt to the LLM server and return the response.
    
    Parameters:
        server_url (str): URL of the LLM server
        payload (dict): Model configuration parameters and generation settings
        repo_path (str): Path to Git repository
        
    Returns:
        str: LLM response content
    """
    logger.debug(f"server_url: {server_url} | payload: {payload}")

    # Get system prompt
    system_prompt = open(os.path.join("data", "prompts", "system", "diff_analyzer.xml")).read()     
    
    # Get diff from the repository
    logger.info(f"Getting diff from repository: {repo_path}")
    diff_content = get_repo_diff(repo_path)
    
    # Save diff content to output.txt
    if diff_content and diff_content != "No changes detected in the repository.":
        save_diff_to_file(diff_content)
        print(f"Diff content saved to output.txt")
    
    # Return if no diff content
    if not diff_content:
        logger.info("No diff content found - skipping prompt")
        return

    try:
        # Set up OpenAI client
        client = openai.OpenAI(
            base_url=f"{server_url}/v1",
            api_key="sk-no-key-required"
        )   

        # Create chat completion request
        completion = client.chat.completions.create(
            model="gemma-3-1b-it-Q4_K_M.gguf",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": diff_content}
            ],
            temperature=payload["generation_settings"]["temperature"],
            top_p=payload["generation_settings"]["top_p"],
            max_tokens=payload["generation_settings"]["n_predict"]
        )

        response = completion.choices[0].message.content
        formatted_response = json.dumps({"response": response}, indent=2)
        
        logger.debug(f"response received | content length: {len(formatted_response)}")
        return formatted_response

    except Exception as e:
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

def handle_hotkey_press(server_url, payload, repo_path):
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
        # Send the prompt to the server
        response = send_prompt_to_server(server_url, payload, repo_path)
        print("\n--- LLM Response ---")
        print(response)
        print("-------------------\n")
    except Exception as e:
        logger.error(f"Error handling hotkey press: {e}")
        print(f"Error: {e}")

def handle_commit_hotkey(repo_path, server_url=None, payload=None):
    """
    Callback function for commit hotkey (Ctrl+Space) press event.
    
    Parameters:
        repo_path (str): Path to Git repository
        server_url (str, optional): URL of the LLM server for generating commit messages
        payload (dict, optional): Model configuration parameters
        
    Returns:
        None
    """
    logger.info("Commit hotkey (Ctrl+Space) pressed - processing git commit")
    
    if not os.path.exists(repo_path):
        error_msg = f"Repository path not found: {repo_path}"
        logger.error(error_msg)
        print(f"\n--- Git Commit Error ---\n{error_msg}\n------------------------\n")
        return
    
    try:
        # Get the diff content
        diff_content = get_repo_diff(repo_path)
        if not diff_content:
            logger.warning("Failed to get repository diff")
            print("\n--- Git Commit Error ---\nFailed to get repository diff\n------------------------\n")
            return
        
        if diff_content == "No changes detected in the repository.":
            print("\n--- Git Commit ---\nNo changes to commit\n------------------\n")
            return
        
        # Generate commit message
        commit_message = "Auto-commit from LLM feedback system"
        llm_message_success = False
        
        # If server_url and payload are provided, try to generate a better commit message using LLM
        if server_url and payload:
            try:
                logger.info("Generating commit message with LLM")
                # Get system prompt for commit message generation
                system_prompt_path = os.path.join("data", "prompts", "system", "diff_analyzer.txt")
                if os.path.exists(system_prompt_path):
                    system_prompt = open(system_prompt_path).read()
                else:
                    # Fallback system prompt if file doesn't exist
                    logger.warning(f"System prompt file not found: {system_prompt_path}")
                
                # Set up OpenAI client
                client = openai.OpenAI(
                    base_url=f"{server_url}/v1",
                    api_key="sk-no-key-required"
                )
                
                # Create chat completion request for commit message
                completion = client.chat.completions.create(
                    model="gemma-3-1b-it-Q4_K_M.gguf",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": diff_content}
                    ],
                    temperature=0,
                    max_tokens=50
                )
                
                generated_message = completion.choices[0].message.content.strip()
                if generated_message:
                    commit_message = generated_message
                    llm_message_success = True
                    logger.info(f"Generated commit message with LLM: {commit_message}")
                else:
                    logger.warning("LLM returned empty commit message, using default")
            except Exception as e:
                logger.warning(f"Failed to generate commit message with LLM: {e}. Using default message.")
                print(f"Note: Failed to generate commit message with LLM: {str(e)[:100]}...")
        
        # Commit the changes
        logger.info(f"Committing changes with message: {commit_message}")
        result = commit_changes(repo_path, commit_message)
        
        print("\n--- Git Commit ---")
        if llm_message_success:
            print(f"Commit Message (LLM-generated): {commit_message}")
        else:
            print(f"Commit Message (default): {commit_message}")
        print(result)
        print("------------------\n")
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error handling commit hotkey press: {error_message}")
        print(f"\n--- Git Commit Error ---\n{error_message}\n------------------------\n")

def setup_keyboard_listener(server_url, payload, repo_path, hotkey=DEFAULT_HOTKEY):
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
        # Create a callback that includes the server_url and payload
        def hotkey_callback():
            handle_hotkey_press(server_url, payload, repo_path)
        
        # Register the hotkey
        keyboard.add_hotkey(hotkey, hotkey_callback)
        logger.info(f"Keyboard listener set up for hotkey: {hotkey}")
        
        # Register the commit hotkey (Ctrl+Space)
        def commit_hotkey_callback():
            handle_commit_hotkey(repo_path, server_url, payload)
        
        keyboard.add_hotkey('ctrl+space', commit_hotkey_callback)
        logger.info("Keyboard listener set up for commit hotkey: ctrl+space")
        
        print(f"Press {hotkey} to get LLM feedback (Ctrl+C to exit)")
        print(f"Press Ctrl+Space to commit changes")
    
    except Exception as e:
        logger.error(f"Error setting up keyboard listener: {e}")
        raise RuntimeError(f"Failed to set up keyboard listener: {e}") 