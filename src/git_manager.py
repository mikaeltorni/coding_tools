"""
git_manager.py

Manages Git repository operations for the LLM feedback system.

Functions:
    get_repo_diff(repo_path): Gets the latest diff from a Git repository in XML format
    is_git_repo(repo_path): Checks if the path is a valid Git repository
"""
import logging
import os
from git import Repo, GitCommandError, InvalidGitRepositoryError

# Configure logging
logger = logging.getLogger(__name__)

def is_git_repo(repo_path):
    """
    Check if the given path is a valid Git repository.
    
    Parameters:
        repo_path (str): Path to the Git repository
        
    Returns:
        bool: True if the path is a valid Git repository, False otherwise
    """
    logger.debug(f"Checking if path is a Git repository: {repo_path}")
    
    try:
        # Try to instantiate a Repo object
        Repo(repo_path)
        logger.debug(f"Valid Git repository found at: {repo_path}")
        return True
    except InvalidGitRepositoryError:
        logger.warning(f"Not a valid Git repository: {repo_path}")
        return False
    except Exception as e:
        logger.error(f"Error checking Git repository: {e}")
        return False

def get_repo_diff(repo_path):
    """
    Get the diff of uncommitted changes in the Git repository and format in XML.
    
    Parameters:
        repo_path (str): Path to the Git repository
        
    Returns:
        str: XML formatted diff content
    """
    logger.debug(f"Getting diff from repository: {repo_path}")
    
    if not is_git_repo(repo_path):
        logger.error(f"Cannot get diff: Not a valid Git repository: {repo_path}")
        error_msg = f"<e>Not a valid Git repository: {repo_path}</e>"
        logger.debug(f"Returning error XML: {error_msg}")
        return error_msg
    
    try:
        logger.debug(f"Initializing repository object for {repo_path}")
        repo = Repo(repo_path)
        
        has_changes = False
        xml_output = "<diffs>\n"  # Start XML document
        diff_id = 1
        
        # Get modified (unstaged) files
        logger.debug("Checking for unstaged modifications")
        unstaged_diffs = list(repo.index.diff(None))
        logger.debug(f"Found {len(unstaged_diffs)} unstaged modifications")
        
        for item in unstaged_diffs:
            has_changes = True
            file_path = item.a_path
            logger.debug(f"Processing unstaged file: {file_path}")
            
            try:
                file_diff = repo.git.diff(None, file_path)
                logger.debug(f"Got diff for {file_path}, length: {len(file_diff)}")
                
                # Format according to expected XML structure
                xml_fragment = f'<diff id="{diff_id}" file="{file_path}" status="modified">{file_diff}</diff>\n'
                logger.debug(f"Created XML fragment for {file_path}, id: {diff_id}, fragment length: {len(xml_fragment)}")
                xml_output += xml_fragment
                diff_id += 1
            except Exception as e:
                logger.error(f"Error getting diff for file {file_path}: {e}")
        
        # Get staged files
        logger.debug("Checking for staged modifications")
        staged_diffs = list(repo.index.diff("HEAD"))
        logger.debug(f"Found {len(staged_diffs)} staged modifications")
        
        for item in staged_diffs:
            has_changes = True
            file_path = item.a_path
            logger.debug(f"Processing staged file: {file_path}")
            
            try:
                file_diff = repo.git.diff("--staged", file_path)
                logger.debug(f"Got diff for staged {file_path}, length: {len(file_diff)}")
                
                # Format according to expected XML structure
                xml_fragment = f'<diff id="{diff_id}" file="{file_path}" status="staged">{file_diff}</diff>\n'
                logger.debug(f"Created XML fragment for staged {file_path}, id: {diff_id}, fragment length: {len(xml_fragment)}")
                xml_output += xml_fragment
                diff_id += 1
            except Exception as e:
                logger.error(f"Error getting diff for staged file {file_path}: {e}")
        
        # Get untracked files
        logger.debug("Checking for untracked files")
        untracked_files = repo.untracked_files
        logger.debug(f"Found {len(untracked_files)} untracked files")
        
        for file_path in untracked_files:
            logger.debug(f"Processing untracked file: {file_path}")
            try:
                full_path = os.path.join(repo_path, file_path)
                logger.debug(f"Reading untracked file from {full_path}")
                with open(full_path, 'r') as f:
                    file_content = f.read()
                
                logger.debug(f"Read content from {file_path}, length: {len(file_content)}")
                has_changes = True
                
                # Format according to expected XML structure
                xml_fragment = f'<diff id="{diff_id}" file="{file_path}" status="untracked">{file_content}</diff>\n'
                logger.debug(f"Created XML fragment for untracked {file_path}, id: {diff_id}, fragment length: {len(xml_fragment)}")
                xml_output += xml_fragment
                diff_id += 1
            except Exception as e:
                logger.error(f"Error reading untracked file {file_path}: {e}")
        
        # If no changes were found
        if not has_changes:
            logger.info("No changes found in the repository")
            error_msg = "<e>No changes detected in the repository.</e>"
            logger.debug(f"Returning no changes XML: {error_msg}")
            return error_msg
        
        xml_output += "</diffs>"  # Close XML document
        
        logger.debug(f"Successfully generated XML diff | total length: {len(xml_output)}")
        logger.debug(f"First 500 characters of XML: {xml_output[:500]}...")
        logger.debug(f"Last 500 characters of XML: {xml_output[-500:] if len(xml_output) > 500 else xml_output}")
        return xml_output
    
    except GitCommandError as e:
        logger.error(f"Git command error getting diff: {e}")
        # Format Git command errors in a more user-friendly way
        error_msg = str(e)
        logger.debug(f"Git command error details: {error_msg}")
        formatted_error = f"<e>Error getting diff: {error_msg}</e>"
        logger.debug(f"Returning Git error XML: {formatted_error}")
        return formatted_error
    except Exception as e:
        logger.error(f"Unexpected error getting diff: {e}")
        logger.debug(f"Exception type: {type(e).__name__}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        formatted_error = f"<e>Error getting diff: {e}</e>"
        logger.debug(f"Returning error XML: {formatted_error}")
        return formatted_error

def get_repo_status(repo_path):
    """
    Get the status of the Git repository.
    
    Parameters:
        repo_path (str): Path to the Git repository
        
    Returns:
        str: Status information as a string
    """
    logger.debug(f"Getting status from repository: {repo_path}")
    
    if not is_git_repo(repo_path):
        logger.error(f"Cannot get status: Not a valid Git repository: {repo_path}")
        return ""
    
    try:
        repo = Repo(repo_path)
        status = repo.git.status()
        logger.debug(f"Successfully retrieved status | length: {len(status)}")
        return status
    
    except GitCommandError as e:
        logger.error(f"Git command error getting status: {e}")
        return f"Error getting status: {e}"
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return f"Error getting status: {e}" 
