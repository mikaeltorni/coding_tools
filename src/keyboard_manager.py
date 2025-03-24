"""
keyboard_manager.py

Manages keyboard hotkey detection and handling for LLM feedback.

Functions:
    send_prompt_to_server(server_url, prompt, model_args): Sends a prompt to the LLM server
    handle_hotkey_press(server_url, model_args): Handles the hotkey press event
    setup_keyboard_listener(server_url, model_args, hotkey): Sets up keyboard listener
"""
import logging
import keyboard
import openai
import os
import json
import datetime
import traceback
from pathlib import Path

from data.model_config import (
    DEFAULT_HOTKEY
)
from src.git_manager import get_repo_diff

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
    system_prompt_path = os.path.join("data", "prompts", "system", "diff_analyzer.xml")
    logger.debug(f"Loading system prompt from: {system_prompt_path}")
    try:
        with open(system_prompt_path, 'r') as f:
            system_prompt = f.read()
        logger.debug(f"System prompt loaded, length: {len(system_prompt)}")
        logger.debug(f"System prompt content: {system_prompt}")
    except Exception as e:
        logger.error(f"Error loading system prompt: {e}")
        print(f"Error loading system prompt: {e}")
        return
    
    # Get diff from the repository
    logger.info(f"Getting diff from repository: {repo_path}")
    diff_content = get_repo_diff(repo_path)
    logger.debug(f"Received diff content, type: {type(diff_content)}, length: {len(diff_content)}")
    logger.debug(f"Diff content: {diff_content}")
    
    # Check if diff is an error message (starts with <e>)
    is_error_diff = diff_content.startswith("<e>") if diff_content else True
    logger.debug(f"Is error diff: {is_error_diff}")
    
    # Save diff content to output.txt and debug file
    if diff_content:
        # Always save to debug file for troubleshooting
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        debug_file = f"diff_debug_{timestamp}.xml"
        
        logger.debug(f"Saving diff content to debug file: {debug_file}")
        try:
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(diff_content)
            logger.debug(f"Debug diff content saved to {debug_file}")
        except Exception as e:
            logger.error(f"Error saving debug diff: {e}")
        
        # Only save non-error diffs to output.txt
        if not is_error_diff:
            logger.debug("Saving diff content to output.txt as it contains changes")
            save_diff_to_file(diff_content)
            print(f"Diff content saved to output.txt")
        else:
            logger.debug(f"Not saving to output.txt as it appears to be an error message: {diff_content[:100]}...")
    else:
        logger.info("No diff content found - skipping prompt")
        return "No diff content found. Please check the repository."

    try:
        # Set up OpenAI client
        logger.debug(f"Initializing OpenAI client with base URL: {server_url}/v1")
        client = openai.OpenAI(
            base_url=f"{server_url}/v1",
            api_key="sk-no-key-required"
        )   

        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": diff_content}
        ]
        logger.debug(f"Prepared messages for completion, system prompt length: {len(messages[0]['content'])}, user prompt length: {len(messages[1]['content'])}")
        
        # Log model parameters
        logger.debug(f"Using model parameters: temperature={payload['generation_settings']['temperature']}, top_p={payload['generation_settings']['top_p']}, max_tokens={payload['generation_settings']['n_predict']}")

        # Save the exact request to a debug file for troubleshooting
        try:
            request_debug = {
                "model": "gemma-3-1b-it-Q4_K_M.gguf",
                "messages": messages,
                "temperature": payload["generation_settings"]["temperature"],
                "top_p": payload["generation_settings"]["top_p"],
                "max_tokens": payload["generation_settings"]["n_predict"]
            }
            request_debug_file = f"request_debug_{timestamp}.json"
            logger.debug(f"Saving request details to debug file: {request_debug_file}")
            with open(request_debug_file, 'w', encoding='utf-8') as f:
                json.dump(request_debug, f, indent=2)
            logger.debug(f"Request debug saved to {request_debug_file}")
        except Exception as e:
            logger.error(f"Error saving request debug: {e}")

        # Create chat completion request
        logger.debug("Sending chat completion request to server")
        
        # Use the best available model - try a larger model if available
        model_name = "llama3"  # Default to a more capable model like llama3 if available
        fallback_model = "gemma-3-1b-it-Q4_K_M.gguf"  # Fallback to gemma if necessary
        
        try:
            # List available models to see what's best to use
            models_response = client.models.list()
            available_models = [model.id for model in models_response.data]
            logger.debug(f"Available models: {available_models}")
            
            # Try to use a better model if available
            preferred_models = ["llama3", "llama-3-70b", "llama-3", "mistral", "gemma-7b", "yi-large"]
            for preferred in preferred_models:
                if any(preferred in model_id.lower() for model_id in available_models):
                    model_matches = [m for m in available_models if preferred in m.lower()]
                    model_name = model_matches[0]  # Use the first match
                    logger.debug(f"Selected preferred model: {model_name}")
                    break
            else:
                # If no preferred model found, use the fallback
                model_name = fallback_model
                logger.debug(f"Using fallback model: {model_name}")
        except Exception as e:
            logger.warning(f"Error listing models, using fallback model: {e}")
            model_name = fallback_model
        
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,  # Lower temperature for more focused reasoning
            top_p=payload["generation_settings"]["top_p"],
            max_tokens=payload["generation_settings"]["n_predict"]
        )

        # Process the response
        response = completion.choices[0].message.content
        logger.debug(f"Received response, length: {len(response)}")
        logger.debug(f"Response content: {response[:500]}...")
        
        formatted_response = json.dumps({"response": response}, indent=2)
        logger.debug(f"Formatted response for output, length: {len(formatted_response)}")
        
        # Save the raw response to a debug file
        try:
            response_debug_file = f"response_debug_{timestamp}.json"
            logger.debug(f"Saving raw response to debug file: {response_debug_file}")
            with open(response_debug_file, 'w', encoding='utf-8') as f:
                f.write(formatted_response)
            logger.debug(f"Response debug saved to {response_debug_file}")
        except Exception as e:
            logger.error(f"Error saving response debug: {e}")
        
        return formatted_response

    except Exception as e:
        logger.error(f"Error sending request to server: {e}")
        logger.debug(f"Exception type: {type(e).__name__}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
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
        
        # Also save to a debug file with timestamp for reference
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        debug_file = f"diff_debug_{timestamp}.xml"
        
        logger.debug(f"Saving diff content to debug file: {debug_file}")
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(diff_content)
        logger.debug(f"Debug diff content saved to {debug_file}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving diff content to file: {e}")
        return False

def handle_hotkey_press(server_url, payload, repo_path):
    """
    Callback function for hotkey press event.
    
    Parameters:
        server_url (str): URL of the LLM server
        payload (dict): Model configuration parameters and generation settings
        repo_path (str): Path to Git repository
        
    Returns:
        None
    """
    logger.info("Hotkey pressed - sending prompt to LLM")
    
    try:                
        # Send the prompt to the server
        print("\nSending request to LLM, please wait...")
        logger.debug(f"Starting prompt send to server with repository path: {repo_path}")
        
        response = send_prompt_to_server(server_url, payload, repo_path)
        
        if response:
            print("\n--- LLM Response ---")
            print(response)
            print("-------------------\n")
            
            # Save response to debug file
            debug_file = "llm_response_debug.json"
            logger.debug(f"Saving LLM response to debug file: {debug_file}")
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(response)
            logger.debug(f"LLM response saved to {debug_file}")
        else:
            logger.warning("No response received from LLM")
            print("\nNo response received from LLM. Check the logs for details.")
    except Exception as e:
        logger.error(f"Error handling hotkey press: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        print(f"\nError: {e}")
        print("Check the logs for detailed debug information.")

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
        print(f"Press {hotkey} to get LLM feedback (Ctrl+C to exit)")
    
    except Exception as e:
        logger.error(f"Error setting up keyboard listener: {e}")
        raise RuntimeError(f"Failed to set up keyboard listener: {e}") 