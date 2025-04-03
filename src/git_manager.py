"""
git_manager.py

Manages Git repository operations for the LLM feedback system.

Functions:
    get_repo_diff(repo_path): Gets the latest diff from a Git repository
    is_git_repo(repo_path): Checks if the path is a valid Git repository
    commit_changes(repo_path, commit_message): Commits all staged changes to the Git repository
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
    Get the diff of uncommitted changes in the Git repository.
    
    Parameters:
        repo_path (str): Path to the Git repository
        
    Returns:
        str: Diff content as string, or empty string if no changes or error occurs
    """
    logger.debug(f"Getting diff from repository: {repo_path}")
    
    if not is_git_repo(repo_path):
        logger.error(f"Cannot get diff: Not a valid Git repository: {repo_path}")
        return ""
    
    try:
        repo = Repo(repo_path)
        
        # Get the diff for all changes (including unstaged)
        diff = repo.git.diff(None)

        logger.debug(f"Diff: {diff}")
        
        # If there are no unstaged changes, check for staged changes
        if not diff:
            diff = repo.git.diff('--staged')
            if diff:
                logger.info("No unstaged changes, but found staged changes")
            else:
                logger.info("No changes (unstaged or staged) found in the repository")
                return "No changes detected in the repository."
        
        logger.debug(f"Successfully retrieved diff | length: {len(diff)}")
        return diff
    
    except GitCommandError as e:
        logger.error(f"Git command error getting diff: {e}")
        return f"Error getting diff: {e}"
    except Exception as e:
        logger.error(f"Error getting diff: {e}")
        return f"Error getting diff: {e}"

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

def commit_changes(repo_path, commit_message):
    """
    Commit all staged changes to the Git repository.
    
    Parameters:
        repo_path (str): Path to the Git repository
        commit_message (str): Commit message
        
    Returns:
        str: Result of the commit operation
    """
    logger.debug(f"Committing changes to repository: {repo_path} | message: {commit_message}")
    
    if not is_git_repo(repo_path):
        error_msg = f"Cannot commit: Not a valid Git repository: {repo_path}"
        logger.error(error_msg)
        return error_msg
    
    try:
        repo = Repo(repo_path)
        
        # Check if there are staged changes
        staged_changes = repo.git.diff('--staged')
        if not staged_changes:
            # No staged changes, so stage all changes
            logger.info("No staged changes found. Staging all changes.")
            repo.git.add('--all')
        
        # Check if there are any changes to commit after staging
        if not repo.git.diff('--staged'):
            logger.info("No changes to commit")
            return "No changes to commit"
        
        # Commit the changes
        # Git will use credentials from environment variables (GIT_AUTHOR_NAME, GIT_AUTHOR_EMAIL)
        # or from the git config if they are set
        result = repo.git.commit('-m', commit_message)
        logger.info(f"Successfully committed changes: {result}")
        return f"Successfully committed changes: {result}"
    
    except GitCommandError as e:
        logger.error(f"Git command error during commit: {e}")
        return f"Error committing changes: {e}"
    except Exception as e:
        logger.error(f"Error committing changes: {e}")
        return f"Error committing changes: {e}" 