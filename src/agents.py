"""
Agents module for different LLM-based agents.

This module defines different agent types that can process
and respond to specific types of inputs.

Classes:
    Agent: Base class for all agents
    DiffReceiver: Agent that processes Git diffs
"""
import logging
import os
import time
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class Agent:
    """
    Base class for all agents.
    """
    
    def __init__(self, name, model, conversation_manager):
        """
        Initialize a new agent.
        
        Parameters:
            name (str): The name of the agent
            model (tuple): (model, config) tuple from ModelManager
            conversation_manager (ConversationManager): Manager for conversations
        """
        self.name = name
        self.model, self.model_config = model
        self.conversation_manager = conversation_manager
        logger.debug(f"Agent '{name}' initialized")
    
    def load_system_prompt(self, prompt_path):
        """
        Loads the system prompt from a file.
        
        Parameters:
            prompt_path (str): Path to the prompt file
            
        Returns:
            str: The system prompt content
        """
        try:
            path = Path(prompt_path)
            if not path.exists():
                logger.error(f"System prompt file not found: {prompt_path}")
                return "You are a helpful assistant."
                
            with open(path, 'r', encoding='utf-8') as file:
                prompt = file.read()
                logger.debug(f"Loaded system prompt from {prompt_path}")
                return prompt
        except Exception as e:
            logger.error(f"Error loading system prompt: {e}")
            return "You are a helpful assistant."
    
    def send_message(self, messages):
        """
        Sends messages to the LLM and returns the response.
        
        Parameters:
            messages (list): List of message dictionaries to send
            
        Returns:
            str: Response from the LLM
        """
        logger.debug(f"Agent '{self.name}' sending messages to model: {len(messages)} messages")
        
        try:
            # Convert messages to a prompt string for llama.cpp
            prompt = self._format_messages_for_llamacpp(messages)
            
            # Send the message to the model
            output = self.model.create_completion(
                prompt,
                max_tokens=self.model_config["max_tokens"],
                temperature=self.model_config["temperature"],
                top_p=self.model_config["top_p"],
                top_k=self.model_config.get("top_k", 40),
                repeat_penalty=self.model_config.get("repeat_penalty", 1.1)
            )
            
            # Get the response
            response = output["choices"][0]["text"]
            
            return response
        except Exception as e:
            logger.error(f"Error sending message to model: {e}")
            return f"Error: Could not get a response from the model: {e}"
    
    def _format_messages_for_llamacpp(self, messages):
        """
        Format messages for the llama.cpp model.
        
        Parameters:
            messages (list): List of message dictionaries
            
        Returns:
            str: Formatted prompt string
        """
        prompt = ""
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            # Handle different content formats
            if isinstance(content, list):
                # If content is a list of message parts, concatenate them
                formatted_content = ""
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        formatted_content += item["text"]
                    elif isinstance(item, str):
                        formatted_content += item
                content = formatted_content
            
            if role == "system":
                prompt += f"<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n\n"
            elif role == "user":
                # If we had a previous system message without a closing tag
                if prompt.endswith("\n\n") and "<</SYS>>" in prompt and not "[/INST]" in prompt:
                    prompt += f"{content} [/INST]\n"
                else:
                    prompt += f"<s>[INST] {content} [/INST]\n"
            elif role == "assistant":
                # Add the assistant's response
                prompt += f"{content}</s>\n"
        
        # If prompt doesn't end with a user message expecting a response, 
        # we need to add a closing tag to the last message
        if not prompt.endswith("[/INST]\n"):
            prompt += " [/INST]\n"
            
        logger.debug(f"Formatted prompt for llama.cpp: {prompt[:100]}...")
        return prompt
    
    def process(self, input_text):
        """
        Process input text and generate a response.
        
        Parameters:
            input_text (str): The input text to process
            
        Returns:
            str: The agent's response
        """
        # This method should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement process()")


class DiffReceiver(Agent):
    """
    Agent that processes Git diffs.
    """
    
    def __init__(self, model, conversation_manager):
        """
        Initialize a new DiffReceiver agent.
        
        Parameters:
            model (tuple): (model, config) tuple from ModelManager
            conversation_manager (ConversationManager): Manager for conversations
        """
        super().__init__("DiffReceiver", model, conversation_manager)
        
        # Load system prompt
        prompt_path = os.path.join("data", "prompts", "system", "diff_analyzer.xml")
        system_prompt = self.load_system_prompt(prompt_path)
        
        # Initialize conversation with system prompt
        self.conversation_manager.init_with_system_prompt(system_prompt)
    
    def process_diffs(self, diffs, file_diffs):
        """
        Process Git diffs and generate a response.
        
        Parameters:
            diffs (str): XML representation of Git diffs
            file_diffs (dict): Dictionary mapping filenames to their diff content
            
        Returns:
            str: The agent's response
        """
        logger.debug("DiffReceiver processing diffs")
        
        # Update file contexts
        self.conversation_manager.update_file_contexts(file_diffs)
        
        # Log the diffs being sent to the model
        logger.debug(f"Diffs being sent to model:\n{diffs}")
        
        # Format the message to ensure diffs are visible
        message_content = f"Here are the current changes in my Git repository:\n\n```xml\n{diffs}\n```\n\nPlease analyze these changes and provide feedback."
        
        # Add user message to conversation
        self.conversation_manager.update_conversation("user", message_content)
        
        # Send message and get response
        messages = self.conversation_manager.get_conversation_history()
        response = self.send_message(messages)
        
        # Add assistant response to conversation
        self.conversation_manager.update_conversation("assistant", response)
        
        return response 