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
import time

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
                    
                    # Start timing
                    start_time = time.time()
                    
                    # Send the request to the server
                    try:
                        response = requests.post(f"{self.server_url}/completion", json=payload)
                        response.raise_for_status()
                        result = response.json()
                        
                        # End timing
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        
                        # Extract the content from the response
                        content = result.get("content", "")
                        
                        # Get timing information from the server if available
                        timings = result.get("timings", {})
                        prompt_eval_time = timings.get("prompt_eval", {}).get("total_ms", 0) / 1000
                        eval_time = timings.get("eval", {}).get("total_ms", 0) / 1000
                        total_time = prompt_eval_time + eval_time
                        
                        # Get token counts from the server if available
                        prompt_tokens = timings.get("prompt_eval", {}).get("n_tokens", 0)
                        completion_tokens = timings.get("eval", {}).get("n_tokens", 0)
                        total_tokens = prompt_tokens + completion_tokens
                        
                        # If server doesn't provide timing info, use our estimates
                        if total_time <= 0:
                            total_time = elapsed_time
                        
                        if total_tokens <= 0:
                            # Estimate token count if not available from server
                            prompt_tokens = len(prompt.split()) * 1.3  # Rough estimate
                            completion_tokens = len(content.split()) * 1.3  # Rough estimate
                            total_tokens = prompt_tokens + completion_tokens
                        
                        # Calculate tokens per second
                        tokens_per_second = completion_tokens / eval_time if eval_time > 0 else 0
                        
                        # Convert the result to a standardized format
                        formatted_result = {
                            "choices": [
                                {
                                    "text": content,
                                    "finish_reason": "stop"
                                }
                            ],
                            "usage": {
                                "prompt_tokens": int(prompt_tokens),
                                "completion_tokens": int(completion_tokens),
                                "total_tokens": int(total_tokens)
                            },
                            "performance": {
                                "prompt_eval_time": prompt_eval_time,
                                "eval_time": eval_time,
                                "total_time": total_time,
                                "tokens_per_second": tokens_per_second
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
                            },
                            "performance": {
                                "prompt_eval_time": 0,
                                "eval_time": 0,
                                "total_time": 0,
                                "tokens_per_second": 0
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