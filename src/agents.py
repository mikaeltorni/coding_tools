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
from pathlib import Path

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
            model (tuple): (pipeline, config) tuple from ModelManager
            conversation_manager (ConversationManager): Manager for conversations
        """
        self.name = name
        self.model_pipe, self.model_config = model
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
            output = self.model_pipe(
                text_inputs=messages, 
                max_new_tokens=self.model_config["max_new_tokens"], 
                temperature=self.model_config["temperature"]
            )
            response = output[0]["generated_text"][-1]["content"]
            logger.debug(f"Agent '{self.name}' received response from model")
            return response
        except Exception as e:
            logger.error(f"Error sending message to model: {e}")
            return f"Error: Could not get a response from the model: {e}"
    
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
            model (tuple): (pipeline, config) tuple from ModelManager
            conversation_manager (ConversationManager): Manager for conversations
        """
        super().__init__("DiffReceiver", model, conversation_manager)
        
        # Load system prompt
        prompt_path = os.path.join("data", "prompts", "diff_receiver.md")
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