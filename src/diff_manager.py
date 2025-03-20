"""
Diff Manager module for handling Git operations.

This module handles Git operations including retrieving diffs.

Classes:
    DiffManager: Manages Git operations and retrieving diffs
    DiffFormatter: Formats diffs into structured output
"""
import logging
from git import Repo

logger = logging.getLogger(__name__)

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
    
    def get_unstaged_changes(self, repo, diff_id_start=1):
        """
        Get unstaged changes from the repository.
        
        Parameters:
            repo (git.Repo): The Git repository object
            diff_id_start (int): Starting ID for the diffs
            
        Returns:
            tuple: (list of diff items, dict of file diffs, next diff ID)
        """
        logger.debug(f"Getting unstaged changes starting with diff ID: {diff_id_start}")
        
        diff_items = []
        file_diffs = {}
        diff_id = diff_id_start
        
        unstaged_changes = list(repo.index.diff(None))
        logger.debug(f"Found {len(unstaged_changes)} unstaged changes")
        
        for diff_item in unstaged_changes:
            try:
                file_path = diff_item.a_path
                logger.debug(f"Processing unstaged change for file: {file_path}")
                
                file_diff = repo.git.diff(file_path)
                item = {
                    'id': diff_id,
                    'file': file_path,
                    'content': file_diff,
                    'status': 'modified'
                }
                diff_items.append(item)
                file_diffs[file_path] = file_diff
                diff_id += 1
                logger.debug(f"Added unstaged change as diff #{diff_id-1}")
                
                # Display diff details
                logger.debug(f"Unstaged change in {file_path}:")
                lines = file_diff.split("\n")[:5]  # Show first 5 lines
                for line in lines:
                    logger.debug(f"  {line}")
                if len(file_diff.split("\n")) > 5:
                    logger.debug("  ...")
            except Exception as e:
                logger.error(f"Error getting diff for {diff_item.a_path}: {e}")
                
        logger.debug(f"Completed unstaged changes processing, next diff ID: {diff_id}")
        return diff_items, file_diffs, diff_id
    
    def get_staged_changes(self, repo, diff_id_start):
        """
        Get staged changes from the repository.
        
        Parameters:
            repo (git.Repo): The Git repository object
            diff_id_start (int): Starting ID for the diffs
            
        Returns:
            tuple: (list of diff items, dict of file diffs, next diff ID)
        """
        logger.debug(f"Getting staged changes starting with diff ID: {diff_id_start}")
        
        diff_items = []
        file_diffs = {}
        diff_id = diff_id_start
        
        staged_changes = list(repo.index.diff('HEAD'))
        logger.debug(f"Found {len(staged_changes)} staged changes")
        
        for diff_item in staged_changes:
            try:
                file_path = diff_item.a_path
                logger.debug(f"Processing staged change for file: {file_path}")
                
                file_diff = repo.git.diff('--cached', file_path)
                item = {
                    'id': diff_id,
                    'file': file_path,
                    'content': file_diff,
                    'status': 'staged'
                }
                diff_items.append(item)
                file_diffs[f"{file_path} (staged)"] = file_diff
                diff_id += 1
                logger.debug(f"Added staged change as diff #{diff_id-1}")
                
                # Display diff details
                logger.debug(f"Staged change in {file_path}:")
                lines = file_diff.split("\n")[:5]  # Show first 5 lines
                for line in lines:
                    logger.debug(f"  {line}")
                if len(file_diff.split("\n")) > 5:
                    logger.debug("  ...")
            except Exception as e:
                logger.error(f"Error getting staged diff for {diff_item.a_path}: {e}")
                
        logger.debug(f"Completed staged changes processing, next diff ID: {diff_id}")
        return diff_items, file_diffs, diff_id
    
    def get_untracked_files(self, repo, diff_id_start):
        """
        Get untracked files from the repository.
        
        Parameters:
            repo (git.Repo): The Git repository object
            diff_id_start (int): Starting ID for the diffs
            
        Returns:
            tuple: (list of diff items, dict of file diffs, next diff ID)
        """
        logger.debug(f"Getting untracked files starting with diff ID: {diff_id_start}")
        
        untracked = repo.untracked_files
        logger.debug(f"Found {len(untracked)} untracked files")
        
        if not untracked:
            logger.debug("No untracked files found")
            return [], {}, diff_id_start
        
        diff_items = []
        file_diffs = {}
        diff_id = diff_id_start
        
        # Process each untracked file individually
        for file_path in untracked:
            try:
                full_path = f"{self.repo_path}/{file_path}"
                # Try to read as text first
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Format content to be more similar to git diff for new files
                        formatted_content = f"New file: {file_path}\n\n" + "\n".join([f"+ {line}" for line in content.split('\n')])
                        
                        item = {
                            'id': diff_id,
                            'file': file_path,
                            'content': formatted_content,
                            'status': 'untracked'
                        }
                        diff_items.append(item)
                        file_diffs[file_path] = content
                        diff_id += 1
                        logger.debug(f"Added untracked file as diff #{diff_id-1}")
                except UnicodeDecodeError:
                    # If not text, mark as binary
                    logger.debug(f"File {file_path} appears to be binary")
                    file_diff = f"New binary file: {file_path}"
                    
                    item = {
                        'id': diff_id,
                        'file': file_path,
                        'content': f"Binary file: {file_path}\n+ [Binary file contents not shown]",
                        'status': 'untracked'
                    }
                    diff_items.append(item)
                    file_diffs[file_path] = "[Binary file]"
                    diff_id += 1
                    logger.debug(f"Added untracked binary file as diff #{diff_id-1}")
            except Exception as e:
                logger.error(f"Error reading untracked file {file_path}: {e}")
                file_diff = f"Error reading file: {file_path}\n{e}"
                
                item = {
                    'id': diff_id,
                    'file': file_path,
                    'content': f"Error reading file: {file_path}\n+ [Error: {e}]",
                    'status': 'untracked'
                }
                diff_items.append(item)
                file_diffs[file_path] = f"[Error reading file: {e}]"
                diff_id += 1
                logger.debug(f"Added untracked file with error as diff #{diff_id-1}")
            
            # Display content preview
            if file_path in file_diffs:
                content_preview = file_diffs[file_path][:100] if isinstance(file_diffs[file_path], str) else str(file_diffs[file_path])
                if len(content_preview) > 100:
                    content_preview = content_preview[:100] + "..."
                logger.debug(f"  Content preview for {file_path}: {content_preview}")
                
        logger.debug(f"Completed untracked files processing, next diff ID: {diff_id}")
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
            diff_items = []
            file_diffs = {}
            
            # Get unstaged changes
            logger.debug("Retrieving unstaged changes")
            unstaged_items, unstaged_diffs, next_id = self.get_unstaged_changes(repo)
            diff_items.extend(unstaged_items)
            file_diffs.update(unstaged_diffs)
            logger.debug(f"Added {len(unstaged_items)} unstaged changes")
            
            # Get staged changes
            logger.debug("Retrieving staged changes")
            staged_items, staged_diffs, next_id = self.get_staged_changes(repo, next_id)
            diff_items.extend(staged_items)
            file_diffs.update(staged_diffs)
            logger.debug(f"Added {len(staged_items)} staged changes")
            
            # Get untracked files
            logger.debug("Retrieving untracked files")
            untracked_items, untracked_diffs, next_id = self.get_untracked_files(repo, next_id)
            if untracked_items:
                diff_items.extend(untracked_items)
                file_diffs.update(untracked_diffs)
                logger.debug(f"Added {len(untracked_items)} untracked files")
            
            # Handle the case where the repository is dirty but no specific diffs were found
            if not diff_items:
                logger.warning("Repository is marked as dirty but no diffs were found")
                return None, {}
            
            # Format the diffs as XML
            logger.debug(f"Formatting {len(diff_items)} diff items as XML")
            xml_output = DiffFormatter.format_to_xml(diff_items)
            logger.info(f"Diff output generated with {len(diff_items)} changes")
            
            # Display diffs in human-readable format
            for item in diff_items:
                if 'file' in item:
                    logger.info(f"Diff #{item['id']} - {item['file']} ({item['status']})")
                else:
                    logger.info(f"Diff #{item['id']} - {item['status']}")
                logger.info("-" * 80)
                # Log the complete diff content
                logger.info(item['content'])
                logger.info("-" * 80)
                logger.info("")
            
            # Log the complete XML payload being sent to the model
            logger.info("XML payload sent to model:")
            logger.info(xml_output)
            
            return xml_output, file_diffs
            
        except Exception as e:
            logger.error(f"Error accessing Git repository: {e}")
            return f"<error>Error accessing Git repository: {e}</error>", {} 
