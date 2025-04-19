"""
keyboard_manager.py

Manages keyboard hotkey detection and handling for LLM feedback.

Functions:
    send_prompt_to_server(server_url, prompt, model_args): Sends a prompt to the LLM server
    handle_hotkey_press(server_url, model_args): Handles the hotkey press event
    handle_commit_hotkey(repo_paths): Handles the commit hotkey (Ctrl+Space) press event
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

def send_prompt_to_server(server_url, payload, repo_paths):
    """
    Send a prompt to the LLM server and return the response.
    
    Parameters:
        server_url (str): URL of the LLM server
        payload (dict): Model configuration parameters and generation settings
        repo_paths (list): List of paths to Git repositories
        
    Returns:
        dict: Dictionary mapping repository paths to LLM responses
    """
    logger.debug(f"server_url: {server_url} | payload: {payload}")
    
    # Get system prompt
    system_prompt = open(os.path.join("data", "prompts", "system", "diff_analyzer.xml")).read()
    
    all_responses = {}
    
    for repo_path in repo_paths:
        logger.info(f"Processing repository: {repo_path}")
        
        # Get diff from the repository
        logger.info(f"Getting diff from repository: {repo_path}")
        diff_content = get_repo_diff(repo_path)
        
        # Save diff content to output file for current repository
        repo_name = os.path.basename(os.path.normpath(repo_path))
        output_file = f"output_{repo_name}.txt"
        if diff_content and diff_content != "No changes detected in the repository.":
            save_diff_to_file(diff_content, output_file)
            print(f"Diff content saved to {output_file}")
        
        # Skip if no diff content
        if not diff_content:
            logger.info(f"No diff content found in {repo_path} - skipping prompt")
            all_responses[repo_path] = "No changes detected in the repository."
            continue
            
        if diff_content == "No changes detected in the repository.":
            all_responses[repo_path] = diff_content
            continue

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
            
            logger.debug(f"Response received for {repo_path} | content length: {len(formatted_response)}")
            all_responses[repo_path] = formatted_response

        except Exception as e:
            error_msg = f"Error sending request to server for {repo_path}: {e}"
            logger.error(error_msg)
            all_responses[repo_path] = error_msg
    
    return all_responses

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

def handle_hotkey_press(server_url, payload, repo_paths):
    """
    Callback function for hotkey press event.
    
    Parameters:
        server_url (str): URL of the LLM server
        payload (dict): Model configuration parameters
        repo_paths (list): List of paths to Git repositories
        
    Returns:
        None
    """
    logger.info(f"Hotkey pressed - processing {len(repo_paths)} repositories")
    
    try:                
        # Send the prompt to the server for each repository
        responses = send_prompt_to_server(server_url, payload, repo_paths)
        
        for repo_path, response in responses.items():
            repo_name = os.path.basename(os.path.normpath(repo_path))
            print(f"\n--- LLM Response for {repo_name} ---")
            print(response)
            print("-------------------\n")
    except Exception as e:
        logger.error(f"Error handling hotkey press: {e}")
        print(f"Error: {e}")

def handle_commit_hotkey(repo_paths, server_url=None, payload=None):
    """
    Callback function for commit hotkey (Ctrl+Space) press event.
    
    Parameters:
        repo_paths (list): List of paths to Git repositories
        server_url (str, optional): URL of the LLM server for generating commit messages
        payload (dict, optional): Model configuration parameters
        
    Returns:
        None
    """
    logger.info(f"Commit hotkey (Ctrl+Space) pressed - processing git commit for {len(repo_paths)} repositories")
    
    for repo_path in repo_paths:
        repo_name = os.path.basename(os.path.normpath(repo_path))
        logger.info(f"Processing repository: {repo_path}")
        
        if not os.path.exists(repo_path):
            error_msg = f"Repository path not found: {repo_path}"
            logger.error(error_msg)
            print(f"\n--- Git Commit Error for {repo_name} ---\n{error_msg}\n------------------------\n")
            continue
        
        try:
            # Get the diff content
            diff_content = get_repo_diff(repo_path)
            if not diff_content:
                logger.warning(f"Failed to get repository diff for {repo_path}")
                print(f"\n--- Git Commit Error for {repo_name} ---\nFailed to get repository diff\n------------------------\n")
                continue
            
            if diff_content == "No changes detected in the repository.":
                print(f"\n--- Git Commit for {repo_name} ---\nNo changes to commit\n------------------\n")
                continue
            
            # Generate commit message
            commit_message = f"Auto-commit from LLM feedback system for {repo_name}"
            llm_message_success = False
            
            # If server_url and payload are provided, try to generate a better commit message using LLM
            if server_url and payload:
                try:
                    logger.info(f"Generating commit message with LLM for {repo_path}")
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
                        max_tokens=4096
                    )
                    
                    generated_message = completion.choices[0].message.content.strip()
                    if generated_message:
                        commit_message = generated_message
                        llm_message_success = True
                        logger.info(f"Generated commit message with LLM for {repo_path}: {commit_message}")
                    else:
                        logger.warning(f"LLM returned empty commit message for {repo_path}, using default")
                except Exception as e:
                    logger.warning(f"Failed to generate commit message with LLM for {repo_path}: {e}. Using default message.")
                    print(f"Note: Failed to generate commit message with LLM for {repo_name}: {str(e)[:100]}...")
            
            # Commit the changes
            logger.info(f"Committing changes in {repo_path} with message: {commit_message}")
            result = commit_changes(repo_path, commit_message)
            
            print(f"\n--- Git Commit for {repo_name} ---")
            if llm_message_success:
                print(f"Commit Message (LLM-generated): {commit_message}")
            else:
                print(f"Commit Message (default): {commit_message}")
            print(result)
            print("------------------\n")
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error handling commit hotkey press for {repo_path}: {error_message}")
            print(f"\n--- Git Commit Error for {repo_name} ---\n{error_message}\n------------------------\n")

def setup_keyboard_listener(server_url, payload, repo_paths, hotkey=DEFAULT_HOTKEY):
    """
    Set up keyboard listener for the specified hotkey.
    
    Parameters:
        server_url (str): URL of the LLM server
        payload (dict): Model configuration parameters
        repo_paths (list): List of paths to Git repositories
        hotkey (str): Keyboard hotkey combination to trigger LLM prompt
        
    Returns:
        None
    """
    logger.debug(f"Setting up keyboard listener for hotkey: {hotkey}")
    
    try:
        # Create a callback that includes the server_url and payload
        def hotkey_callback():
            handle_hotkey_press(server_url, payload, repo_paths)
        
        # Register the hotkey
        keyboard.add_hotkey(hotkey, hotkey_callback)
        logger.info(f"Keyboard listener set up for hotkey: {hotkey}")
        
        # Register the commit hotkey (Ctrl+Space)
        def commit_hotkey_callback():
            handle_commit_hotkey(repo_paths, server_url, payload)
        
        keyboard.add_hotkey('ctrl+space', commit_hotkey_callback)
        logger.info("Keyboard listener set up for commit hotkey: ctrl+space")
        
    except Exception as e:
        logger.error(f"Error setting up keyboard listener: {e}")
        raise RuntimeError(f"Failed to setup keyboard listener: {e}") 