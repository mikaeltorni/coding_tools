"""
Models module for connecting to LLM servers.

This module handles the connection to external LLM servers through REST APIs.

Classes:
    ModelManager: Manages connections to LLM servers
"""
import logging
import requests
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages connections to LLM servers.
    """
    
    @staticmethod
    def create_server_client(server_url="http://localhost:8080", config=None):
        """
        Creates a client for connecting to a LLM server.
        
        Parameters:
            server_url (str): URL of the LLM server
            config (dict, optional): Configuration parameters for model generation
            
        Returns:
            tuple: (server_client, config) - Server client object and its configuration
        """
        logger.debug(f"Creating LLM server client for server at {server_url}")
        
        # Default configuration with optimized settings
        default_config = {
            "max_tokens": 4000,  # Maximum number of tokens to generate
            "top_p": 0.9,        # Top-p sampling
            "top_k": 40,         # Top-k sampling
            "temperature": 0.01, # Temperature for sampling
            "repeat_penalty": 1.1, # Penalty for repeated tokens
            "n_predict": 4000    # Number of tokens to predict (server-specific)
        }
        
        # Use provided config or default, with provided values overriding defaults
        merged_config = default_config.copy()
        if config:
            merged_config.update(config)
        
        try:
            # Verify server is accessible
            try:
                response = requests.get(f"{server_url}/health")
                if response.status_code != 200:
                    logger.warning(f"Server health check failed with status {response.status_code}")
            except requests.RequestException as e:
                logger.warning(f"Could not connect to LLM server: {e}")
                logger.warning("Continuing anyway, as the server might be available later")
                
            # Create a server client class
            class LlamaServerClient:
                def __init__(self, server_url, config):
                    self.server_url = server_url
                    self.config = config
                
                def create_completion(self, prompt, **kwargs):
                    """
                    Creates a completion by sending a request to the LLM server.
                    
                    Parameters:
                        prompt (str): The prompt to complete
                        **kwargs: Additional parameters to override config
                        
                    Returns:
                        dict: The completion response
                    """
                    logger.debug("Sending completion request to LLM server")
                    
                    # Merge config with kwargs
                    request_config = self.config.copy()
                    request_config.update(kwargs)
                    
                    # Prepare the payload
                    payload = {
                        "prompt": prompt,
                        "n_predict": request_config.get("max_tokens", request_config.get("n_predict", 4000)),
                        "temperature": request_config.get("temperature", 0.01),
                        "top_p": request_config.get("top_p", 0.9),
                        "top_k": request_config.get("top_k", 40),
                        "repeat_penalty": request_config.get("repeat_penalty", 1.1)
                    }
                    
                    # Send the request to the server
                    try:
                        response = requests.post(f"{self.server_url}/completion", json=payload)
                        response.raise_for_status()
                        result = response.json()
                        
                        # Convert the result to a standardized format
                        formatted_result = {
                            "choices": [
                                {
                                    "text": result.get("content", ""),
                                    "finish_reason": "stop"
                                }
                            ],
                            "usage": {
                                "prompt_tokens": len(prompt) // 4,  # Rough estimate
                                "completion_tokens": len(result.get("content", "")) // 4,  # Rough estimate
                                "total_tokens": (len(prompt) + len(result.get("content", ""))) // 4  # Rough estimate
                            }
                        }
                        
                        return formatted_result
                    except Exception as e:
                        logger.error(f"Error in server request: {e}")
                        # Return a minimal response format on error
                        return {
                            "choices": [
                                {
                                    "text": f"Error communicating with LLM server: {e}",
                                    "finish_reason": "error"
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0
                            }
                        }
            
            # Create the server client
            server_client = LlamaServerClient(server_url, merged_config)
            
            logger.debug("LLM server client created successfully")
            # Explicitly return a tuple of (server_client, config)
            return (server_client, merged_config)
            
        except Exception as e:
            logger.error(f"Error creating LLM server client: {e}")
            raise RuntimeError(f"Failed to create LLM server client: {e}") 