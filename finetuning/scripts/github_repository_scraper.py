"""
github_repository_scraper.py

Scrapes a GitHub repository including all branches and integrates with Gemma 3 model
for analyzing Git diffs.

Functions:
    clone_repository(repo_url, target_dir): Clones a GitHub repository
    fetch_all_branches(repo): Fetches all available branches
    checkout_branch(repo, branch_name): Checks out a specific branch
    get_branch_diff(repo, branch_name, base_branch): Gets diff between branches
    get_commit_diff(repo, commit): Gets diff for a specific commit
    get_branch_commits(repo, branch_name, after_date=None): Gets all commits from a branch with optional date filtering
    analyze_diff(diff_content, server_url, payload): Analyzes diff using Gemma model
    analyze_diff_manually(diff_content): Fallback method for diff analysis when LLM server fails
    generate_alpaca_dataset(instruction, input_text, output_text): Generates dataset in Alpaca format
    get_diff_hash(diff): Creates a hash for a diff
    main(): Main function that runs the repository scraper

Command Line Usage Examples:
    python github_repository_scraper.py https://github.com/username/repo.git
    python github_repository_scraper.py https://github.com/username/repo.git --target-dir ./repos
    python github_repository_scraper.py https://github.com/username/repo.git --server-url http://localhost:8080
    python github_repository_scraper.py https://github.com/username/repo.git --after-date 2023-01-01
    python github_repository_scraper.py https://github.com/username/repo.git --max-commits 100
    python github_repository_scraper.py https://github.com/username/repo.git --skip-branches "gh-pages,docs"
    python github_repository_scraper.py https://github.com/username/repo.git --process-all-branches
"""
import argparse
import json
import logging
import os
import shutil
import sys
import datetime
from pathlib import Path
import openai
from git import Repo, GitCommandError, InvalidGitRepositoryError, Remote

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s:%(funcName)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Default values
DEFAULT_TARGET_DIR = "./cloned_repos"
DEFAULT_SERVER_URL = "http://localhost:8080"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 4096

SYSTEM_PROMPT = "You are an expert at analyzing Git diffs and classifying changes in short, 10-15 word summaries. Read the provided Git diff and classify it with one of the following tags: feat: A new feature, fix: A bug fix, docs: Documentation only changes, style: Changes that do not affect the meaning of the code, refactor: A code change that neither fixes a bug nor adds a feature, perf: A code change that improves performance, test: Adding missing tests or correcting existing tests, build: Changes that affect the build system or external dependencies, ci: Changes to CI configuration files and scripts, chore: Other changes that don't modify src or test files. You can also use these tags with scopes in parentheses to provide more context, for example: fix(deps): Update dependency versions, feat(auth): Add new authentication method. Your response should be a short 10-15 word summary starting with the tag. For example: 'feat: implemented user authentication with JWT tokens' or 'fix(deps): updated npm dependencies to fix security vulnerabilities'. By any means, do not exceed the 15 word limit."

def clone_repository(repo_url, target_dir):
    """
    Clone a GitHub repository to a local directory.
    
    Parameters:
        repo_url (str): URL of the GitHub repository
        target_dir (str): Target directory to clone into
        
    Returns:
        git.Repo: Cloned repository object
    """
    logger.debug(f"repo_url: {repo_url} | target_dir: {target_dir}")
    
    try:
        # Extract repository name from URL
        repo_name = repo_url.split('/')[-1]
        if repo_name.endswith('.git'):
            repo_name = repo_name[:-4]
        
        # Create target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        
        # Use a timestamp to create a unique directory name
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        repo_path = os.path.join(target_dir, f"{repo_name}_{timestamp}")
        
        # Clone the repository
        logger.info(f"Cloning repository from {repo_url} to {repo_path}")
        repo = Repo.clone_from(repo_url, repo_path)
        logger.info(f"Repository cloned successfully to {repo_path}")
        
        return repo
    
    except GitCommandError as e:
        logger.error(f"Git command error cloning repository: {e}")
        raise RuntimeError(f"Failed to clone repository: {e}")
    except Exception as e:
        logger.error(f"Error cloning repository: {e}")
        raise RuntimeError(f"Failed to clone repository: {e}")

def fetch_all_branches(repo):
    """
    Fetch all available branches from the remote repository.
    
    Parameters:
        repo (git.Repo): Repository object
        
    Returns:
        list: List of branch names
    """
    logger.debug(f"Fetching all branches for repository")
    
    try:
        # Fetch all branches from remote
        logger.info("Fetching all branches from remote")
        for remote in repo.remotes:
            remote.fetch()
        
        # Get list of remote branches
        remote_branches = []
        for ref in repo.remote().refs:
            branch_name = ref.name.split('/')[-1]
            if branch_name != 'HEAD':
                remote_branches.append(branch_name)
        
        logger.info(f"Found {len(remote_branches)} remote branches: {remote_branches}")
        return remote_branches
    
    except GitCommandError as e:
        logger.error(f"Git command error fetching branches: {e}")
        raise RuntimeError(f"Failed to fetch branches: {e}")
    except Exception as e:
        logger.error(f"Error fetching branches: {e}")
        raise RuntimeError(f"Failed to fetch branches: {e}")

def checkout_branch(repo, branch_name):
    """
    Checkout a specific branch in the repository.
    
    Parameters:
        repo (git.Repo): Repository object
        branch_name (str): Name of the branch to checkout
        
    Returns:
        bool: True if checkout successful, False otherwise
    """
    logger.debug(f"branch_name: {branch_name}")
    
    try:
        # Check if branch exists locally
        local_branch_exists = branch_name in [b.name for b in repo.branches]
        
        if local_branch_exists:
            # Checkout local branch
            logger.info(f"Checking out local branch: {branch_name}")
            repo.git.checkout(branch_name)
        else:
            # Create local branch tracking remote
            logger.info(f"Creating local branch from remote: {branch_name}")
            repo.git.checkout('-b', branch_name, f'origin/{branch_name}')
        
        current_branch = repo.active_branch.name
        logger.info(f"Current branch: {current_branch}")
        
        return True
    
    except GitCommandError as e:
        logger.error(f"Git command error checking out branch: {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking out branch: {e}")
        return False

def get_branch_diff(repo, branch_name, base_branch="main"):
    """
    Get the diff between the current branch and the base branch.
    
    Parameters:
        repo (git.Repo): Repository object
        branch_name (str): Name of the branch to compare
        base_branch (str): Name of the base branch to compare against
        
    Returns:
        str: Diff content as string
    """
    logger.debug(f"Getting diff between {branch_name} and {base_branch}")
    
    try:
        if branch_name == base_branch:
            # For the base branch, get the diff from the last commit
            last_commit = list(repo.iter_commits(max_count=1))[0]
            before_last = f"{last_commit.hexsha}~1"
            diff = repo.git.diff(before_last, last_commit.hexsha)
        else:
            # Get diff between branches
            diff = repo.git.diff(base_branch, branch_name)
        
        logger.debug(f"Successfully retrieved diff | length: {len(diff)}")
        
        # If diff is empty, return a message
        if not diff:
            return f"No changes detected between {branch_name} and {base_branch}."
        
        return diff
    
    except GitCommandError as e:
        logger.error(f"Git command error getting branch diff: {e}")
        return f"Error getting diff: {e}"
    except Exception as e:
        logger.error(f"Error getting branch diff: {e}")
        return f"Error getting diff: {e}"

def get_commit_diff(repo, commit):
    """
    Get the diff for a specific commit.
    
    Parameters:
        repo (git.Repo): Repository object
        commit (git.Commit): Commit object to get diff for
        
    Returns:
        str: Diff content as string
    """
    logger.debug(f"Getting diff for commit: {commit.hexsha[:8]}")
    
    try:
        # Get parent commit
        parents = commit.parents
        if not parents:
            # For first commit with no parent
            diff = repo.git.show(commit.hexsha)
        else:
            # Get diff between commit and its first parent
            parent = parents[0]
            diff = repo.git.diff(parent.hexsha, commit.hexsha)
        
        logger.debug(f"Successfully retrieved commit diff | length: {len(diff)}")
        
        # If diff is empty, return a message
        if not diff:
            return f"No changes detected in commit {commit.hexsha[:8]}."
        
        return diff
    
    except GitCommandError as e:
        logger.error(f"Git command error getting commit diff: {e}")
        return f"Error getting diff: {e}"
    except Exception as e:
        logger.error(f"Error getting commit diff: {e}")
        return f"Error getting diff: {e}"

def get_branch_commits(repo, branch_name, after_date=None):
    """
    Get all commits from a branch.
    
    Parameters:
        repo (git.Repo): Repository object
        branch_name (str): Name of the branch to get commits from
        after_date (str): Only include commits after this date (format: YYYY-MM-DD)
        
    Returns:
        list: List of git.Commit objects
    """
    logger.debug(f"Getting commits for branch: {branch_name} | after_date: {after_date}")
    
    try:
        # Get list of commits for the branch
        if after_date:
            try:
                # Use git's date format for filtering
                commits = list(repo.iter_commits(branch_name, since=after_date))
                logger.info(f"Filtering commits after {after_date}")
            except ValueError as e:
                logger.error(f"Invalid date format: {e}. Use YYYY-MM-DD format.")
                commits = list(repo.iter_commits(branch_name))
        else:
            commits = list(repo.iter_commits(branch_name))
        
        logger.info(f"Found {len(commits)} commits in branch {branch_name}")
        return commits
    
    except GitCommandError as e:
        logger.error(f"Git command error getting commits: {e}")
        return []
    except Exception as e:
        logger.error(f"Error getting commits: {e}")
        return []

def analyze_diff(diff_content, server_url, payload):
    """
    Analyze Git diff using Gemma model to classify changes.
    
    Parameters:
        diff_content (str): Git diff content
        server_url (str): URL of the LLM server
        payload (dict): Model configuration parameters
        
    Returns:
        str: LLM response content
    """
    logger.debug(f"server_url: {server_url} | payload: {payload}")
    
    # Skip if no diff content
    if not diff_content or diff_content.startswith("No changes detected"):
        logger.info("No diff content to analyze")
        return "chore: No changes to analyze"

    try:
        # Set up OpenAI client
        client = openai.OpenAI(
            base_url=f"{server_url}/v1",
            api_key="sk-no-key-required",
            timeout=60.0  # Add timeout to avoid hanging
        )   

        # Create chat completion request
        completion = client.chat.completions.create(
            model="gemma-3-4b-it",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": diff_content}
            ],
            temperature=payload["generation_settings"]["temperature"],
            top_p=payload["generation_settings"]["top_p"],
            max_tokens=payload["generation_settings"]["n_predict"]
        )

        response = completion.choices[0].message.content.strip()
        logger.debug(f"Response received: {response}")
        
        # Ensure response is not empty
        if not response:
            logger.warning("Empty response received from model, falling back to manual analysis")
            return analyze_diff_manually(diff_content)
            
        # Ensure response starts with a conventional commit tag (with or without scope)
        import re
        if not re.match(r'^(feat|fix|docs|style|refactor|perf|test|build|ci|chore)(\([a-z0-9-_]+\))?:', response.lower()):
            logger.warning(f"Response doesn't start with a tag: {response}")
            # Add default tag based on diff content
            return "chore: " + response
            
        return response

    except openai.APITimeoutError:
        logger.error("Request to LLama server timed out")
        return "refactor: Component restructured for better maintenance"  # Fallback response
    except openai.APIConnectionError:
        logger.error("Connection error to LLama server")
        # Analyze diff manually as fallback
        return analyze_diff_manually(diff_content)
    except Exception as e:
        logger.error(f"Error sending request to server: {e}")
        return analyze_diff_manually(diff_content)

def analyze_diff_manually(diff_content):
    """
    Manually analyze Git diff when LLama server is unavailable.
    Implements a simple fallback mechanism to determine diff type.
    
    Parameters:
        diff_content (str): Git diff content
        
    Returns:
        str: Classification of the diff
    """
    logger.debug("Using manual diff analysis as fallback")
    
    # Count added and removed lines
    added_lines = diff_content.count('\n+')
    removed_lines = diff_content.count('\n-')
    
    # Check for documentation changes
    if 'README' in diff_content or 'documentation' in diff_content.lower() or '.md' in diff_content:
        return "docs: Updated documentation files or comments"
    
    # Check for test changes
    if '/test/' in diff_content or 'test_' in diff_content or '_test' in diff_content:
        return "test: Updated test files or test configuration"
    
    # Check for dependency changes
    if 'package.json' in diff_content and ('version' in diff_content or 'dependency' in diff_content):
        return "fix(deps): Updated package dependencies to newer versions"
    elif 'requirements.txt' in diff_content or 'Gemfile' in diff_content or 'pyproject.toml' in diff_content:
        return "fix(deps): Updated project dependencies"
    elif any(dep in diff_content.lower() for dep in ['dependency', 'dependencies', 'npm', 'pip', 'yarn', 'composer']):
        return "fix(deps): Updated external dependencies"
    elif 'package-lock.json' in diff_content or 'yarn.lock' in diff_content or 'Gemfile.lock' in diff_content:
        return "fix(deps): Updated dependency lock files"
    
    # Check for CI changes
    if '.github/workflows' in diff_content or '.gitlab-ci' in diff_content or 'jenkins' in diff_content:
        return "ci: Updated CI configuration files"
    
    # Determine if this is a fix or feature based on line count
    if removed_lines > added_lines:
        return "fix: Fixed bug in code implementation"
    else:
        return "feat: Added new functionality to component"

def generate_alpaca_dataset(instruction, input_text, output_text):
    """
    Generate dataset entry in Alpaca format without the text field.
    
    Parameters:
        instruction (str): The instruction for the task
        input_text (str): The input context
        output_text (str): The output response
        
    Returns:
        dict: Dataset entry in Alpaca format
    """
    logger.debug(f"Generating Alpaca dataset entry")
    
    # Create dataset entry without text field
    dataset_entry = {
        "instruction": instruction,
        "input": input_text,
        "output": output_text
    }
    
    return dataset_entry

def get_diff_hash(diff):
    """
    Create a hash for a diff to uniquely identify it.
    
    Parameters:
        diff (str): The diff content
        
    Returns:
        str: MD5 hash of the normalized diff
    """
    import hashlib
    # Use a consistent subset of the diff to create a hash
    # First, normalize line endings
    normalized_diff = diff.replace('\r\n', '\n')
    # Create hash from the normalized content
    return hashlib.md5(normalized_diff.encode('utf-8')).hexdigest()

def is_conventional_commit(commit_message):
    """
    Check if the commit message follows the conventional commit format.
    
    Parameters:
        commit_message (str): The commit message to check
        
    Returns:
        bool: True if the message follows conventional format, False otherwise
    """
    # Get the first line of the commit message
    first_line = commit_message.strip().split('\n')[0].strip()
    
    # List of conventional commit types
    conventional_types = ['feat', 'fix', 'docs', 'style', 'refactor', 
                         'perf', 'test', 'build', 'ci', 'chore']
    
    # Check if the message follows conventional format with or without scope
    import re
    # Pattern to match: type(scope): description or type: description
    pattern = r'^(feat|fix|docs|style|refactor|perf|test|build|ci|chore)(\([a-z0-9-_]+\))?:\s.+'
    
    if re.match(pattern, first_line.lower()):
        return True
    
    return False

def main():
    """
    Main function that parses command line arguments and runs the repository scraper.
    
    Parameters:
        None
        
    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description='Scrape a GitHub repository including all branches and analyze diffs using Gemma model.'
    )
    parser.add_argument('repo_url', type=str, help='URL of the GitHub repository to scrape')
    parser.add_argument('--target-dir', type=str, default=DEFAULT_TARGET_DIR,
                       help=f'Target directory to clone the repository into (default: {DEFAULT_TARGET_DIR})')
    parser.add_argument('--server-url', type=str, default=DEFAULT_SERVER_URL,
                       help=f'URL of the llama server (default: {DEFAULT_SERVER_URL})')
    parser.add_argument('--output-file', type=str, default='github_diff_dataset.json',
                       help='Output file for the Alpaca dataset (default: github_diff_dataset.json)')
    parser.add_argument('--max-diff-size', type=int, default=100000,
                       help='Maximum diff size in characters to process (default: 100000)')
    parser.add_argument('--max-commits', type=int, default=None,
                       help='Maximum number of commits to process per branch (default: all commits)')
    parser.add_argument('--after-date', type=str, default=None,
                       help='Only process commits after this date (format: YYYY-MM-DD)')
    parser.add_argument('--skip-branches', type=str, default=None,
                       help='Comma-separated list of branches to skip (e.g., "gh-pages,docs")')
    parser.add_argument('--process-all-branches', action='store_true',
                       help='Process all branches instead of just the main branch (for testing)')
    
    # Model configuration arguments
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                       help=f'Temperature parameter for text generation (default: {DEFAULT_TEMPERATURE})')
    parser.add_argument('--max-tokens', type=int, default=DEFAULT_MAX_TOKENS,
                       help=f'Maximum number of tokens to generate (default: {DEFAULT_MAX_TOKENS})')
    parser.add_argument('--stop-at', type=str, default=None,
                       help='Stop processing when reaching this commit SHA (e.g., "70051a0c05d03455dd1ad77062006b95f07b8e46")')
    
    try:
        args = parser.parse_args()
        
        repo_url = args.repo_url
        target_dir = args.target_dir
        server_url = args.server_url
        output_file = args.output_file
        max_diff_size = args.max_diff_size
        max_commits = args.max_commits
        after_date = args.after_date
        skip_branches = args.skip_branches
        process_all_branches = args.process_all_branches
        stop_at_commit = args.stop_at
        
        # Set up payload for the LLM
        payload = {
            "generation_settings": {
                "temperature": args.temperature,
                "top_p": DEFAULT_TOP_P,
                "n_predict": args.max_tokens
            }
        }
        
        # Check if output file exists and load existing dataset
        dataset = []
        existing_entries_by_diff = {}  # Track existing entries by diff hash
        
        if os.path.exists(output_file):
            try:
                logger.info(f"Loading existing dataset from {output_file}")
                with open(output_file, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
                logger.info(f"Loaded {len(dataset)} existing entries")
                
                # Index existing entries for quick lookup
                for i, entry in enumerate(dataset):
                    if entry.get("input"):
                        # Get diff hash directly from the input content
                        # This handles both old format (with metadata) and new format (diff only)
                        input_content = entry["input"]
                        diff_start = input_content.find("diff --git")
                        
                        if diff_start != -1:
                            # If it contains "diff --git", use that part for hashing
                            diff_content = input_content[diff_start:]
                            diff_hash = get_diff_hash(diff_content)
                            existing_entries_by_diff[diff_hash] = i
                        else:
                            # If it's already in the new format (just diff), use entire input
                            diff_hash = get_diff_hash(input_content)
                            existing_entries_by_diff[diff_hash] = i
                
                logger.info(f"Indexed {len(existing_entries_by_diff)} entries by diff hash")
                
            except json.JSONDecodeError:
                logger.warning(f"Error loading existing dataset. Starting with empty dataset.")
                dataset = []
            except Exception as e:
                logger.warning(f"Could not load existing file: {e}. Starting with empty dataset.")
                dataset = []
        
        # Clone the repository
        repo = clone_repository(repo_url, target_dir)
        repo_path = repo.working_dir
        
        # Function to save dataset to file
        def save_dataset(new_entry=None):
            if new_entry is not None:
                # Get diff hash for duplicate checking
                diff_hash = get_diff_hash(diff_content)
                
                # Check if already exists by diff hash
                if diff_hash in processed_diffs:
                    logger.info(f"Entry with identical diff already exists in dataset, skipping commit {commit.hexsha[:8]}")
                    return False
                else:
                    # Add to dataset and update indexes
                    dataset.append(new_entry)
                    processed_diffs.add(diff_hash)
                        
            logger.info(f"Saving dataset with {len(dataset)} entries to {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2)
            return True
        
        # Fetch all branches
        branches = fetch_all_branches(repo)
        
        # Process skip-branches parameter
        branches_to_skip = []
        if skip_branches:
            branches_to_skip = [branch.strip() for branch in skip_branches.split(',')]
            logger.info(f"Skipping branches: {branches_to_skip}")
        
        # For testing, only process the main branch
        main_branch_names = ["main", "master"]
        test_branches = [branch for branch in branches if branch in main_branch_names]
        
        if not process_all_branches:
            if test_branches:
                logger.info(f"Testing mode: only processing main branch: {test_branches[0]}")
                branches = [test_branches[0]]
            else:
                logger.warning("Main branch not found in repository. Make sure the main branch is named 'main' or 'master'.")
                if branches:
                    logger.info(f"Using first available branch instead: {branches[0]}")
                    branches = [branches[0]]
        else:
            logger.info(f"Processing all branches: {len(branches)} branches found")
        
        # Track processed diffs to avoid duplicates
        processed_diffs = set(existing_entries_by_diff.keys())
        processed_commits = set()  # Still track processed commits for this run
        
        logger.info(f"Found {len(processed_diffs)} already processed unique diffs")
        
        # Process each branch
        total_new_entries = 0
        for branch_name in branches:
            if branch_name in branches_to_skip:
                logger.info(f"Skipping branch {branch_name} as requested")
                continue
                
            logger.info(f"Processing branch: {branch_name}")
            
            # Checkout branch
            if checkout_branch(repo, branch_name):
                # Get all commits for the branch
                commits = get_branch_commits(repo, branch_name, after_date)
                
                # Limit number of commits if specified
                if max_commits is not None and len(commits) > max_commits:
                    logger.info(f"Limiting to {max_commits} commits (out of {len(commits)})")
                    commits = commits[:max_commits]
                
                # Process each commit
                branch_new_entries = 0
                for commit in commits:
                    # Check if we've reached the stop commit
                    if stop_at_commit and commit.hexsha.startswith(stop_at_commit):
                        logger.info(f"Reached stop-at commit: {commit.hexsha[:8]}, stopping processing")
                        print(f"Reached stop-at commit: {commit.hexsha[:8]}, stopping processing")
                        break
                        
                    logger.info(f"Processing commit: {commit.hexsha[:8]} - {commit.message.splitlines()[0]}")
                    
                    # Get commit diff
                    diff_content = get_commit_diff(repo, commit)
                    
                    # Skip if no diff content
                    if diff_content.startswith("No changes detected") or diff_content.startswith("Error getting diff"):
                        logger.info(f"Skipping commit {commit.hexsha[:8]}: {diff_content}")
                        continue
                    
                    # Check if this diff is already in the dataset
                    diff_hash = get_diff_hash(diff_content)
                    if diff_hash in processed_diffs:
                        logger.info(f"Skipping commit {commit.hexsha[:8]}: Duplicate diff already processed")
                        # Mark this commit as processed to avoid future attempts
                        processed_commits.add(commit.hexsha)
                        continue
                    
                    # Skip if we've already processed this commit in this run
                    if commit.hexsha in processed_commits:
                        logger.info(f"Skipping already processed commit: {commit.hexsha[:8]}")
                        continue
                    
                    # Limit diff size to avoid overwhelming the model
                    if len(diff_content) > max_diff_size:
                        logger.warning(f"Diff content too large ({len(diff_content)} chars), truncating to {max_diff_size} chars")
                        diff_content = diff_content[:max_diff_size] + "\n... (truncated)"
                    
                    # Check if commit message already follows conventional format
                    if is_conventional_commit(commit.message):
                        logger.info(f"Commit {commit.hexsha[:8]} already has conventional format, using original message")
                        # Just use the first line of the commit message
                        analysis = commit.message.strip().split('\n')[0].strip()
                    else:
                        # Analyze diff using Gemma model
                        logger.info(f"Analyzing diff for commit: {commit.hexsha[:8]}")
                        analysis = analyze_diff(diff_content, server_url, payload)
                    
                    # Ensure analysis is not empty
                    if not analysis or analysis.strip() == "":
                        logger.warning("Empty analysis result, using fallback")
                        analysis = "chore: Repository maintenance or minor changes"
                    
                    # Only include the diff content without commit metadata
                    # No need to store commit metadata in the dataset to avoid cleanup later
                    dataset_entry = generate_alpaca_dataset(SYSTEM_PROMPT, diff_content, analysis)
                    
                    # Add to dataset and save immediately
                    if save_dataset(dataset_entry):
                        # Only increment counters if the entry was actually added
                        processed_commits.add(commit.hexsha)
                        branch_new_entries += 1
                        total_new_entries += 1
                        logger.info(f"Commit {commit.hexsha[:8]} processed: {analysis}")
                
                logger.info(f"Branch {branch_name} processing complete. Processed {branch_new_entries} new commits.")
            else:
                logger.warning(f"Failed to checkout branch: {branch_name}")
        
        logger.info(f"Repository processing complete. Added {total_new_entries} new entries to dataset.")
        print(f"Repository processing complete. Dataset saved to {output_file} with {len(dataset)} total entries.")
        
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
        print("\nProgram terminated by user.")
        # Final save at exit
        if 'dataset' in locals() and 'output_file' in locals():
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2)
            print(f"Partial dataset saved to {output_file} with {len(dataset)} entries.")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"Error: {e}")
        # Final save at exit
        if 'dataset' in locals() and 'output_file' in locals():
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2)
            print(f"Partial dataset saved to {output_file} with {len(dataset)} entries despite error.")
        sys.exit(1)

if __name__ == "__main__":
    main() 