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
        Gets the current uncommitted diffs from the Git repository and formats them in XML.
        
        Parameters:
            None
            
        Returns:
            tuple: (str or None, dict) - XML representation of all diffs (or None if no changes) and dictionary mapping filenames to diffs
        """
        logger.debug(f"Getting diffs from: {self.repo_path}")
        
        try:
            repo = Repo(self.repo_path)
            
            # Check both staged and unstaged changes
            has_changes = repo.is_dirty(untracked_files=True)
            
            if not has_changes and not repo.untracked_files:
                logger.info("No changes detected in repository")
                return None, {}
            
            # Get diffs for all modified files
            diffs = []
            file_diffs = {}  # Dictionary to store diffs by filename
            diff_id = 1
            
            # Get unstaged changes (working tree changes)
            for diff_item in repo.index.diff(None):
                try:
                    file_diff = repo.git.diff(diff_item.a_path)
                    xml_diff = f"<diff id=\"{diff_id}\" file=\"{diff_item.a_path}\" status=\"modified\">\n{file_diff}\n</diff>"
                    diffs.append(xml_diff)
                    file_diffs[diff_item.a_path] = file_diff
                    diff_id += 1
                except Exception as e:
                    logger.error(f"Error getting diff for {diff_item.a_path}: {e}")
            
            # Get staged changes (index changes)
            for diff_item in repo.index.diff('HEAD'):
                try:
                    file_diff = repo.git.diff('--cached', diff_item.a_path)
                    xml_diff = f"<diff id=\"{diff_id}\" file=\"{diff_item.a_path}\" status=\"staged\">\n{file_diff}\n</diff>"
                    diffs.append(xml_diff)
                    file_diffs[f"{diff_item.a_path} (staged)"] = file_diff
                    diff_id += 1
                except Exception as e:
                    logger.error(f"Error getting staged diff for {diff_item.a_path}: {e}")
            
            # Add untracked files
            untracked = repo.untracked_files
            if untracked:
                untracked_text = "\n".join(untracked)
                xml_untracked = f"<diff id=\"{diff_id}\" status=\"untracked\">\n{untracked_text}\n</diff>"
                diffs.append(xml_untracked)
                file_diffs["untracked_files"] = untracked_text
            
            if not diffs:
                logger.warning("Repository is marked as dirty but no diffs were found")
                return None, {}
            
            # Wrap all diffs in a root element
            xml_output = "<diffs>\n" + "\n".join(diffs) + "\n</diffs>"
            return xml_output, file_diffs
        except Exception as e:
            logger.error(f"Error accessing Git repository: {e}")
            return f"<error>Error accessing Git repository: {e}</error>", {} 
