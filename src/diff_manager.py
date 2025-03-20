"""
Diff Manager module for handling Git operations.

This module handles Git operations including retrieving diffs.

Classes:
    DiffManager: Manages Git operations and retrieving diffs
"""
import logging
from git import Repo

logger = logging.getLogger(__name__)

class DiffManager:
    """
    Manages Git operations and retrieving diffs.
    """
    
    def __init__(self, repo_path):
        """
        Initialize the DiffManager with a repository path.
        
        Parameters:
            repo_path (str): Path to the Git repository
        """
        self.repo_path = repo_path
        logger.debug(f"DiffManager initialized with repo path: {repo_path}")
    
    def get_diffs(self):
        """
        Gets the current uncommitted diffs from the Git repository.
        
        Parameters:
            None
            
        Returns:
            tuple: (str, dict) - Text representation of all diffs and dictionary mapping filenames to diffs
        """
        logger.debug(f"Getting diffs from: {self.repo_path}")
        
        try:
            repo = Repo(self.repo_path)
            
            # Check both staged and unstaged changes
            has_changes = repo.is_dirty(untracked_files=True)
            
            if not has_changes and not repo.untracked_files:
                logger.info("No changes detected in repository")
                return "No changes detected in repository.", {}
            
            # Get diffs for all modified files
            diffs = []
            file_diffs = {}  # Dictionary to store diffs by filename
            
            # Get unstaged changes (working tree changes)
            for diff_item in repo.index.diff(None):
                try:
                    file_diff = repo.git.diff(diff_item.a_path)
                    diffs.append(f"File: {diff_item.a_path}\n{file_diff}\n")
                    file_diffs[diff_item.a_path] = file_diff
                except Exception as e:
                    logger.error(f"Error getting diff for {diff_item.a_path}: {e}")
            
            # Get staged changes (index changes)
            for diff_item in repo.index.diff('HEAD'):
                try:
                    file_diff = repo.git.diff('--cached', diff_item.a_path)
                    diffs.append(f"File (staged): {diff_item.a_path}\n{file_diff}\n")
                    file_diffs[f"{diff_item.a_path} (staged)"] = file_diff
                except Exception as e:
                    logger.error(f"Error getting staged diff for {diff_item.a_path}: {e}")
            
            # Add untracked files
            untracked = repo.untracked_files
            if untracked:
                untracked_text = "Untracked files:\n" + "\n".join(untracked)
                diffs.append(untracked_text)
                file_diffs["untracked_files"] = untracked_text
            
            if not diffs:
                logger.warning("Repository is marked as dirty but no diffs were found")
                return "Repository has changes, but no specific diffs were detected.", {}
                
            return "\n".join(diffs), file_diffs
        except Exception as e:
            logger.error(f"Error accessing Git repository: {e}")
            return f"Error accessing Git repository: {e}", {} 