"""
Conversation management module.

This module handles conversations with LLM models including
context tracking and message formatting.

Classes:
    ConversationManager: Manages conversation history and message formatting
"""
import logging

logger = logging.getLogger(__name__)

class ConversationManager:
    """
    Manages conversation history and message formatting.
    """
    
    def __init__(self):
        """
        Initialize a new conversation manager.
        
        Parameters:
            None
        """
        self.conversation_history = []
        self.file_contexts = {}  # Stores context for each edited file
        logger.debug("ConversationManager initialized")
    
    def update_conversation(self, role, content):
        """
        Updates the conversation history with a new message.
        
        Parameters:
            role (str): The role of the message (system, user, assistant)
            content (str): The content of the message
            
        Returns:
            list: Updated conversation history
        """
        self.conversation_history.append({
            "role": role,
            "content": [{"type": "text", "text": content}]
        })
        logger.debug(f"Added message with role '{role}' to conversation history")
        return self.conversation_history
    
    def update_file_contexts(self, file_diffs):
        """
        Updates the context for each file that has been edited.
        
        Parameters:
            file_diffs (dict): Dictionary mapping filenames to their diff content
            
        Returns:
            None
        """
        logger.debug(f"Updating context for {len(file_diffs)} files")
        
        for filename, diff in file_diffs.items():
            if filename in self.file_contexts:
                # Update existing context
                self.file_contexts[filename]["history"].append(diff)
                # Keep only the last 5 diffs to prevent context from growing too large
                self.file_contexts[filename]["history"] = self.file_contexts[filename]["history"][-5:]
            else:
                # Create new context
                self.file_contexts[filename] = {
                    "history": [diff],
                    "last_updated": 0
                }
        
        logger.debug(f"File contexts updated. Total tracked files: {len(self.file_contexts)}")
    
    def get_file_context(self, filename):
        """
        Gets context information for a specific file.
        
        Parameters:
            filename (str): The filename to get context for
            
        Returns:
            dict: The file context or None if not found
        """
        return self.file_contexts.get(filename)
    
    def get_conversation_history(self):
        """
        Gets the current conversation history.
        
        Parameters:
            None
            
        Returns:
            list: The current conversation history
        """
        return self.conversation_history
    
    def reset_conversation(self):
        """
        Resets the conversation history.
        
        Parameters:
            None
            
        Returns:
            None
        """
        self.conversation_history = []
        logger.debug("Conversation history reset")
    
    def init_with_system_prompt(self, system_prompt):
        """
        Initializes the conversation with a system prompt.
        
        Parameters:
            system_prompt (str): The system prompt content
            
        Returns:
            None
        """
        if not self.conversation_history:
            self.update_conversation("system", system_prompt)
            logger.debug("Conversation initialized with system prompt")
        else:
            logger.warning("Conversation already has history. Not initializing with system prompt.") 