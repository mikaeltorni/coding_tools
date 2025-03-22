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
        self.is_quantized_model = hasattr(self.model_pipe, 'get_performance_metrics')
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
            # Initial setup phase - don't measure this time
            pipe_args = {
                "text_inputs": messages,
                "max_new_tokens": self.model_config["max_new_tokens"],
                "temperature": self.model_config["temperature"]
            }
            
            # Start timing ONLY after all setup is done
            start_time = time.time()
            
            # Send the message to the model
            output = self.model_pipe(**pipe_args)
            
            # End timing as soon as we get a response
            end_time = time.time()
            
            # Get the response
            response = output[0]["generated_text"][-1]["content"]
            
            # Calculate generation metrics
            elapsed_time = end_time - start_time
            
            # Use a more accurate tokenizer estimate for tokens
            # GPT models estimate: 1 token ≈ 4 characters (including spaces)
            char_count = len(response)
            token_count = char_count // 4
            
            # Alternative count based on words (typically more accurate for English)
            word_count = len(response.split())
            # Most models: 1 token ≈ 0.75 words for English text
            token_count_by_words = int(word_count / 0.75)
            
            # Use the larger of the two estimates to be conservative
            token_count = max(token_count, token_count_by_words)
            
            tokens_per_second = token_count / elapsed_time if elapsed_time > 0 else 0
            
            # Ensure performance metrics are inside the response text, not after any markers
            if "========== END RESPONSE ==========" in response:
                # Split the response and insert performance info before the END marker
                parts = response.split("========== END RESPONSE ==========")
                performance_info = f"\n\n[Performance: {tokens_per_second:.2f} tokens/sec | {token_count} tokens in {elapsed_time:.2f}s]"
                response = f"{parts[0]}{performance_info}\n\n========== END RESPONSE =========={parts[1]}"
            else:
                # Just append performance info to the end of the response
                performance_info = f"\n\n[Performance: {tokens_per_second:.2f} tokens/sec | {token_count} tokens in {elapsed_time:.2f}s]"
                if not response.endswith("]"):  # Avoid adding if already has performance metrics
                    response += performance_info
            
            logger.debug(f"Generation: {token_count} tokens in {elapsed_time:.2f}s = {tokens_per_second:.2f} tokens/sec")
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