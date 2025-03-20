"""
Diff Manager module for handling Git operations.

This module handles Git operations including retrieving diffs.

Classes:
    DiffManager: Manages Git operations and retrieving diffs
    DiffFormatter: Formats diffs into structured output
"""
import logging
from git import Repo
from enum import Enum, auto

logger = logging.getLogger(__name__)

class ChangeType(Enum):
    """Enum for different types of file changes in git repository"""
    UNSTAGED = auto()
    STAGED = auto()
    UNTRACKED = auto()

class DiffFormatter:
    """
    Formats diffs into structured output formats.
    """
    
    @staticmethod
    def format_to_xml(diff_items):
        """
        Formats diff items into XML structure.
        
        Parameters:
            diff_items (list): List of diff item dictionaries
            
        Returns:
            str: XML representation of all diffs
        """
        logger.debug(f"Formatting {len(diff_items) if diff_items else 0} diff items to XML")
        
        if not diff_items:
            logger.debug("No diff items to format, returning None")
            return None
            
        diffs = []
        for item in diff_items:
            diff_id = item['id']
            content = item['content']
            status = item['status']
            
            # Always include the file attribute for all diffs, including untracked files
            if 'file' in item:
                file_path = item['file']
                logger.debug(f"Formatting diff item #{diff_id} for file: {file_path} with status: {status}")
                xml_diff = f"<diff id=\"{diff_id}\" file=\"{file_path}\" status=\"{status}\">\n{content}\n</diff>"
            else:
                # This path should almost never be taken now that we have individual untracked files
                logger.debug(f"Formatting diff item #{diff_id} with status: {status}")
                xml_diff = f"<diff id=\"{diff_id}\" status=\"{status}\">\n{content}\n</diff>"
                
            diffs.append(xml_diff)
            
        # Wrap all diffs in a root element
        xml_output = "<diffs>\n" + "\n".join(diffs) + "\n</diffs>"
        logger.debug(f"XML formatting complete with {len(diffs)} diff elements")
        return xml_output


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
    
    def get_repo(self):
        """
        Get the Git repository object.
        
        Returns:
            git.Repo: The Git repository object
        """
        logger.debug(f"Opening Git repository at: {self.repo_path}")
        try:
            repo = Repo(self.repo_path)
            logger.debug("Repository opened successfully")
            return repo
        except Exception as e:
            logger.error(f"Failed to open repository: {e}")
            raise
    
    def has_changes(self, repo):
        """
        Check if the repository has any changes.
        
        Parameters:
            repo (git.Repo): The Git repository object
            
        Returns:
            bool: True if the repository has changes, False otherwise
        """
        logger.debug("Checking if repository has changes")
        is_dirty = repo.is_dirty(untracked_files=True)
        has_untracked = bool(repo.untracked_files)
        
        logger.debug(f"Repository status - dirty: {is_dirty}, untracked files: {has_untracked}")
        return is_dirty or has_untracked
    
    def _process_file_change(self, file_path, content, status, diff_id):
        """
        Process a file change and create a diff item.
        
        Parameters:
            file_path (str): Path to the changed file
            content (str): Content of the diff
            status (str): Status of the change (modified, staged, or untracked)
            diff_id (int): ID for the diff
            
        Returns:
            tuple: (diff item dict, updated diff ID)
        """
        item = {
            'id': diff_id,
            'file': file_path,
            'content': content,
            'status': status
        }
        
        # Display content preview for logging
        content_preview = content.split("\n")[:5]  # Show first 5 lines
        for line in content_preview:
            logger.debug(f"  {line}")
        if len(content.split("\n")) > 5:
            logger.debug("  ...")
            
        return item, diff_id + 1
    
    def _read_untracked_file(self, file_path):
        """
        Read an untracked file and format its content.
        
        Parameters:
            file_path (str): Path to the untracked file
            
        Returns:
            str: Formatted content of the file
        """
        full_path = f"{self.repo_path}/{file_path}"
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Format content to be more similar to git diff for new files
                formatted_content = f"New file: {file_path}\n\n" + "\n".join([f"+ {line}" for line in content.split('\n')])
                return formatted_content, content
        except UnicodeDecodeError:
            # If not text, mark as binary
            logger.debug(f"File {file_path} appears to be binary")
            return f"Binary file: {file_path}\n+ [Binary file contents not shown]", "[Binary file]"
        except Exception as e:
            logger.error(f"Error reading untracked file {file_path}: {e}")
            return f"Error reading file: {file_path}\n+ [Error: {e}]", f"[Error reading file: {e}]"
    
    def get_changes(self, repo, diff_id_start=1, change_types=None):
        """
        Get changes from the repository of specified types.
        
        Parameters:
            repo (git.Repo): The Git repository object
            diff_id_start (int): Starting ID for the diffs
            change_types (list): List of ChangeType enums to include (default: all types)
            
        Returns:
            tuple: (list of diff items, dict of file diffs, next diff ID)
        """
        if change_types is None:
            change_types = [ChangeType.UNSTAGED, ChangeType.STAGED, ChangeType.UNTRACKED]
            
        logger.debug(f"Getting changes starting with diff ID: {diff_id_start}")
        logger.debug(f"Change types to process: {[t.name for t in change_types]}")
        
        diff_items = []
        file_diffs = {}
        diff_id = diff_id_start
        
        # Process unstaged changes
        if ChangeType.UNSTAGED in change_types:
            logger.debug("Processing unstaged changes")
            unstaged_changes = list(repo.index.diff(None))
            logger.debug(f"Found {len(unstaged_changes)} unstaged changes")
            
            for diff_item in unstaged_changes:
                try:
                    file_path = diff_item.a_path
                    logger.debug(f"Processing unstaged change for file: {file_path}")
                    
                    file_diff = repo.git.diff(file_path)
                    item, diff_id = self._process_file_change(file_path, file_diff, 'modified', diff_id)
                    diff_items.append(item)
                    file_diffs[file_path] = file_diff
                    logger.debug(f"Added unstaged change as diff #{diff_id-1}")
                except Exception as e:
                    logger.error(f"Error getting diff for {diff_item.a_path}: {e}")
        
        # Process staged changes
        if ChangeType.STAGED in change_types:
            logger.debug("Processing staged changes")
            staged_changes = list(repo.index.diff('HEAD'))
            logger.debug(f"Found {len(staged_changes)} staged changes")
            
            for diff_item in staged_changes:
                try:
                    file_path = diff_item.a_path
                    logger.debug(f"Processing staged change for file: {file_path}")
                    
                    file_diff = repo.git.diff('--cached', file_path)
                    item, diff_id = self._process_file_change(file_path, file_diff, 'staged', diff_id)
                    diff_items.append(item)
                    file_diffs[f"{file_path} (staged)"] = file_diff
                    logger.debug(f"Added staged change as diff #{diff_id-1}")
                except Exception as e:
                    logger.error(f"Error getting staged diff for {diff_item.a_path}: {e}")
        
        # Process untracked files
        if ChangeType.UNTRACKED in change_types:
            logger.debug("Processing untracked files")
            untracked = repo.untracked_files
            logger.debug(f"Found {len(untracked)} untracked files")
            
            for file_path in untracked:
                try:
                    logger.debug(f"Processing untracked file: {file_path}")
                    formatted_content, raw_content = self._read_untracked_file(file_path)
                    item, diff_id = self._process_file_change(file_path, formatted_content, 'untracked', diff_id)
                    diff_items.append(item)
                    file_diffs[file_path] = raw_content
                    logger.debug(f"Added untracked file as diff #{diff_id-1}")
                except Exception as e:
                    logger.error(f"Error processing untracked file {file_path}: {e}")
        
        logger.debug(f"Completed changes processing, next diff ID: {diff_id}")
        return diff_items, file_diffs, diff_id
    
    def get_diffs(self):
        """
        Gets the current uncommitted diffs from the Git repository and formats them in XML.
        
        Parameters:
            None
            
        Returns:
            tuple: (str or None, dict) - XML representation of all diffs (or None if no changes) and dictionary mapping filenames to diffs
        """
        logger.debug(f"Getting diffs from repository: {self.repo_path}")
        
        try:
            repo = self.get_repo()
            
            # Check if there are any changes
            if not self.has_changes(repo):
                logger.info("No changes detected in repository")
                return None, {}
            
            # Get all types of changes
            diff_items, file_diffs, _ = self.get_changes(repo)
            
            # Handle the case where the repository is dirty but no specific diffs were found
            if not diff_items:
                logger.warning("Repository is marked as dirty but no diffs were found")
                return None, {}
            
            # Format the diffs as XML
            logger.debug(f"Formatting {len(diff_items)} diff items as XML")
            xml_output = DiffFormatter.format_to_xml(diff_items)
            logger.info(f"Diff output generated with {len(diff_items)} changes")
            
            # Log a summary of changes at INFO level
            for item in diff_items:
                if 'file' in item:
                    logger.info(f"Diff #{item['id']} - {item['file']} ({item['status']})")
                else:
                    logger.info(f"Diff #{item['id']} - {item['status']}")
            
            return xml_output, file_diffs
            
        except Exception as e:
            logger.error(f"Error accessing Git repository: {e}")
            return f"<error>Error accessing Git repository: {e}</error>", {} 
