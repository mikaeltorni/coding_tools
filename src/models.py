"""
Models module for connecting to LLM servers.

This module handles the connection to external LLM servers through REST APIs.

Classes:
    ModelConfig: Configuration class for LLM models
    LlamaServerClient: Client for connecting to llama.cpp HTTP server
    ModelManager: Factory class for creating LLM clients
"""
import logging
import requests
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

logger = logging.getLogger(__name__)

class ModelConfig:
    """
    Configuration class for LLM models.
    """
    
    def __init__(self, 
                 model_type: str = "llama_server",
                 max_tokens: int = 16384,
                 temperature: float = 0.01,
                 top_p: float = 0.9,
                 top_k: int = 40,
                 repeat_penalty: float = 1.1,
                 **kwargs):
        """
        Initialize a new model configuration.
        
        Parameters:
            model_type (str): Type of model ("llama_server", etc.)
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Temperature for sampling
            top_p (float): Top-p sampling parameter
            top_k (int): Top-k sampling parameter
            repeat_penalty (float): Penalty for repeated tokens
            **kwargs: Additional model-specific parameters
        """
        self.model_type = model_type
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty
        
        # Store any additional parameters
        self.extra_params = kwargs
        
        # For the llama server, n_predict is the parameter used instead of max_tokens
        if model_type == "llama_server" and "n_predict" not in kwargs:
            self.extra_params["n_predict"] = max_tokens
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the configuration
        """
        config_dict = {
            "model_type": self.model_type,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty
        }
        
        # Add extra parameters
        config_dict.update(self.extra_params)
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """
        Create a configuration from a dictionary.
        
        Parameters:
            config_dict (Dict[str, Any]): Dictionary with configuration parameters
            
        Returns:
            ModelConfig: New configuration instance
        """
        # Extract known parameters
        model_type = config_dict.pop("model_type", "llama_server")
        max_tokens = config_dict.pop("max_tokens", 4000)
        temperature = config_dict.pop("temperature", 0.01)
        top_p = config_dict.pop("top_p", 0.9)
        top_k = config_dict.pop("top_k", 40)
        repeat_penalty = config_dict.pop("repeat_penalty", 1.1)
        
        # Create with known parameters and pass the rest as extra_params
        return cls(
            model_type=model_type,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            **config_dict
        )


class LlamaServerClient:
    """
    Client for connecting to llama.cpp HTTP server.
    """
    
    def __init__(self, server_url: str, config: ModelConfig):
        """
        Initialize a new llama server client.
        
        Parameters:
            server_url (str): URL of the llama server
            config (ModelConfig): Configuration for the model
        """
        self.server_url = server_url
        self.config = config
    
    def create_completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Creates a completion by sending a request to the llama server.
        
        Parameters:
            prompt (str): The prompt to complete
            **kwargs: Additional parameters to override config
            
        Returns:
            Dict[str, Any]: The completion response
        """
        logger.debug("Sending completion request to llama server")
        
        # Get the config dictionary and update with kwargs
        config_dict = self.config.to_dict()
        config_dict.update(kwargs)
        
        # Prepare the payload for the llama server
        payload = {
            "prompt": prompt,
            "n_predict": config_dict.get("n_predict", config_dict.get("max_tokens", 4000)),
            "temperature": config_dict.get("temperature", 0.01),
            "top_p": config_dict.get("top_p", 0.9),
            "top_k": config_dict.get("top_k", 40),
            "repeat_penalty": config_dict.get("repeat_penalty", 1.1)
        }
        
        # Send the request to the server
        try:
            response = requests.post(f"{self.server_url}/completion", json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Extract the content from the response
            content = result.get("content", "")
            
            # Convert the result to a standardized format
            formatted_result = {
                "choices": [
                    {
                        "text": content,
                        "finish_reason": "stop"
                    }
                ]
            }
            
            return formatted_result
        except Exception as e:
            logger.error(f"Error in server request: {e}")
            # Return a minimal response format on error
            return {
                "choices": [
                    {
                        "text": f"Error communicating with llama server: {e}",
                        "finish_reason": "error"
                    }
                ]
            }


class ModelManager:
    """
    Factory class for creating LLM clients.
    """
    
    @staticmethod
    def create_server_client(server_url: str = "http://localhost:8080", 
                             config: Optional[Union[Dict[str, Any], ModelConfig]] = None) -> Tuple[LlamaServerClient, ModelConfig]:
        """
        Creates a client for connecting to a llama server.
        
        Parameters:
            server_url (str): URL of the llama server
            config (Optional[Union[Dict[str, Any], ModelConfig]]): Configuration for the model
            
        Returns:
            Tuple[LlamaServerClient, ModelConfig]: Server client object and its configuration
        """
        logger.debug(f"Creating llama server client for server at {server_url}")
        
        # Convert dict config to ModelConfig or create a default
        if config is None:
            model_config = ModelConfig(model_type="llama_server")
        elif isinstance(config, dict):
            # Ensure model_type is set
            if "model_type" not in config:
                config["model_type"] = "llama_server"
            model_config = ModelConfig.from_dict(config)
        else:
            model_config = config
        
        try:
            # Verify server is accessible
            try:
                response = requests.get(f"{server_url}/health")
                if response.status_code != 200:
                    logger.warning(f"Server health check failed with status {response.status_code}")
            except requests.RequestException as e:
                logger.warning(f"Could not connect to llama server: {e}")
                logger.warning("Continuing anyway, as the server might be available later")
            
            # Create the server client
            server_client = LlamaServerClient(server_url, model_config)
            
            logger.debug("Llama server client created successfully")
            # Return tuple of (client, config)
            return (server_client, model_config)
        
        except Exception as e:
            logger.error(f"Error creating llama server client: {e}")
            raise RuntimeError(f"Failed to create llama server client: {e}")
    
    @classmethod
    def create_client(cls, client_type: str = "llama_server", **kwargs) -> Tuple[Any, ModelConfig]:
        """
        Creates a client of the specified type.
        
        Parameters:
            client_type (str): Type of client to create
            **kwargs: Arguments for the client creation
            
        Returns:
            Tuple[Any, ModelConfig]: Client instance and its configuration
        """
        if client_type == "llama_server":
            server_url = kwargs.pop("server_url", "http://localhost:8080")
            config = kwargs.pop("config", None)
            return cls.create_server_client(server_url, config)
        else:
            raise ValueError(f"Unknown client type: {client_type}") 