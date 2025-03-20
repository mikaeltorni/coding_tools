"""
Monitor module for keyboard input and event handling.

This module handles keyboard monitoring and event triggers for the application.

Classes:
    KeyMonitor: Monitors keyboard events and triggers actions
"""
import logging
import keyboard

logger = logging.getLogger(__name__)

class KeyMonitor:
    """
    Monitors keyboard events and triggers actions.
    """
    
    def __init__(self, repo_path, diff_manager, agent):
        """
        Initialize a new key monitor.
        
        Parameters:
            repo_path (str): Path to the Git repository
            diff_manager (DiffManager): Manager for Git diffs
            agent (Agent): Agent to process diffs
        """
        self.repo_path = repo_path
        self.diff_manager = diff_manager
        self.agent = agent
        logger.debug(f"KeyMonitor initialized with repo path: {repo_path}")
    
    def start_monitoring(self, hotkey='ctrl+5'):
        """
        Starts monitoring for the specified hotkey.
        
        Parameters:
            hotkey (str): Hotkey combination to monitor for, default is 'ctrl+5'
            
        Returns:
            None
        """
        logger.debug(f"Starting hotkey monitor with hotkey: {hotkey}")
        
        def on_hotkey():
            logger.info(f"Hotkey {hotkey} detected, processing diffs")
            
            try:
                # Get the current diffs
                diffs, file_diffs = self.diff_manager.get_diffs()
                
                if diffs and diffs != "No changes detected in repository.":
                    # Process diffs with the agent
                    response = self.agent.process_diffs(diffs, file_diffs)
                    
                    # Display response
                    print("\n\n========== MODEL RESPONSE ==========\n")
                    print(response)
                    print("\n========== END RESPONSE ==========\n")
                else:
                    print("No changes to process.")
            except Exception as e:
                logger.error(f"Error processing diffs: {e}")
        
        # Register the hotkey
        try:
            keyboard.add_hotkey(hotkey, on_hotkey)
            logger.info(f"Hotkey registered: {hotkey}")
            print(f"Monitoring Git repository. Press {hotkey} to get AI feedback on current changes.")
            print("Press CTRL+C to exit.")
            
            # Keep the program running
            keyboard.wait('ctrl+c')
        except Exception as e:
            logger.error(f"Error setting up hotkey monitoring: {e}")
            raise RuntimeError(f"Failed to set up hotkey monitoring: {e}") 