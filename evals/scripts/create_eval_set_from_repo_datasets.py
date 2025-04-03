"""
create_eval_set_from_repo_datasets.py

Script to create evaluation sets from repository datasets in JSON format.
This script loops through JSON files in the repo_datasets folder and generates
evaluation entries for each item in the datasets.

Functions:
    load_json_datasets(directory): Loads all JSON datasets from a directory
    create_eval_yaml(datasets, output_path, script_dir): Creates evaluation YAML file from datasets
    main(): Main function that orchestrates the evaluation set creation process

Command Line Usage Examples:
    python create_eval_set_from_repo_datasets.py
    python create_eval_set_from_repo_datasets.py --datasets-dir ../../finetuning/repo_datasets
    python create_eval_set_from_repo_datasets.py --output-file ../eval_configs/diff_analyzer_eval_generated.yaml
    python create_eval_set_from_repo_datasets.py --max-entries 50 --max-diff-size 50000
"""
import os
import json
import yaml
import logging
import argparse
import hashlib
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(levelname)s:%(funcName)s: %(message)s'
)
logger = logging.getLogger(__name__)

def load_json_datasets(directory):
    """
    Loads all JSON datasets from a directory.
    
    Parameters:
        directory (str): Path to the directory containing JSON datasets
        
    Returns:
        list: List of loaded datasets
    """
    logger.debug(f"directory: {directory}")
    
    datasets = []
    try:
        # Create path object
        dir_path = Path(directory)
        
        # Check if directory exists
        if not dir_path.exists() or not dir_path.is_dir():
            logger.error(f"Directory not found: {directory}")
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Get all JSON files in the directory
        json_files = list(dir_path.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files in {directory}")
        
        # Load each JSON file
        for json_file in json_files:
            logger.info(f"Loading dataset from {json_file}")
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Validate it's a list
                if not isinstance(data, list):
                    logger.error(f"Dataset {json_file} is not a list, skipping")
                    continue
                    
                logger.info(f"Loaded {len(data)} entries from {json_file.name}")
                
                # Check if any entries are missing "input" or "output" fields
                valid_entries = [e for e in data if "input" in e and "output" in e]
                if len(valid_entries) != len(data):
                    logger.warning(f"Found {len(data) - len(valid_entries)} entries missing input or output in {json_file.name}")
                    
                datasets.append({
                    "file_name": json_file.name,
                    "data": data
                })
                logger.info(f"Successfully loaded {len(data)} entries from {json_file.name}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {json_file}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
                continue
                
        return datasets
        
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise

def generate_hash(content):
    """
    Generate a hash for content to create unique filenames.
    
    Parameters:
        content (str): Content to hash
        
    Returns:
        str: MD5 hash of the content
    """
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def create_eval_yaml(datasets, output_path, script_dir, max_entries=None, max_diff_size=None):
    """
    Creates evaluation YAML file from datasets.
    
    Parameters:
        datasets (list): List of datasets loaded from JSON files
        output_path (str): Path to output the YAML file
        script_dir (Path): Path to the script directory (for resolving relative paths)
        max_entries (int, optional): Maximum number of entries to include per dataset
        max_diff_size (int, optional): Maximum size of diff content in characters
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.debug(f"output_path: {output_path} | max_entries: {max_entries} | max_diff_size: {max_diff_size}")
    
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a directory for diff files
        diffs_dir = output_dir / "diffs"
        diffs_dir.mkdir(exist_ok=True)
        logger.info(f"Created directory for diff files: {diffs_dir}")
        
        # Instead of using YAML library, we'll manually create the YAML with correct formatting
        yaml_header = """description: Diff Analyzer Agent Evals - Auto-generated from repository datasets
prompts:
- '{{message}}'
providers:
- id: llama:gemma-3-1b-it-Q4_K_M
  config:
    temperature: 0
    max_new_tokens: 1024
    top_p: 0.9
    prompt:
      prefix: "<start_of_turn>user\\n"
      suffix: "<end_of_turn>\\n<start_of_turn>model"
    apiEndpoint: ${LLAMA_BASE_URL:-http://localhost:8080}
defaultTest:
  options:
    provider: openai:gpt-4o-mini-2024-07-18
tests:
"""
        
        # Check if assertion prompt file exists
        assertion_file = Path(output_path).parent / ".." / "assertion_prompts" / "diff_analyzer_assertion.md"
        
        if not assertion_file.exists():
            logger.error(f"Assertion prompt file not found at {assertion_file}")
            raise FileNotFoundError(f"Assertion prompt file not found at {assertion_file}")
        
        logger.info(f"Using existing assertion prompt at {assertion_file}")
        
        # Load system prompt content
        system_prompt_path = "../../data/prompts/system/diff_analyzer.txt"
        system_prompt_content = ""
        try:
            system_prompt_file = script_dir / system_prompt_path
            if system_prompt_file.exists():
                with open(system_prompt_file, 'r', encoding='utf-8') as f:
                    system_prompt_content = f.read().strip()
                logger.info(f"Successfully loaded system prompt from {system_prompt_file}")
            else:
                logger.warning(f"System prompt file not found at {system_prompt_file}, using default system prompt")
        except Exception as e:
            logger.error(f"Error reading system prompt: {e}")
            system_prompt_content = "You are an expert at analyzing Git diffs and classifying their changes in short, 10-15 word summaries. Make sure to read the diffs line-by-line for the provided diff by reading what has been added, and removed on the currently unstaged files in the repository. Then proceed to classify it with one of the tags, that are the following: feat: A new feature, fix: A bug fix, docs: Documentation only changes, style: Changes that do not affect the meaning of the code, refactor: A code change that neither fixes a bug nor adds a feature, perf: A code change that improves performance, test: Adding missing tests or correcting existing tests, build: Changes that affect the build system or external dependencies, ci: Changes to CI configuration files and scripts, chore: Other changes that don't modify src or test files. You can also use these tags with scopes in parentheses to provide more context, for example: fix(deps): Update dependency versions, feat(auth): Add new authentication method. Your response should be a short 10-15 word summary starting with the tag. For example: 'feat: implemented user authentication with JWT tokens' or 'fix(deps): updated npm dependencies to fix security vulnerabilities'. By any means, do not exceed the 15 word limit, and do not produce anything more than this one sentence."

        # Track statistics
        stats = {}
        total_entries = 0
        skipped_entries = 0
        
        # Track files created to avoid duplicates
        created_files = set()
        
        # Open the file to write the YAML content
        with open(output_path, 'w', encoding='utf-8') as yaml_file:
            # Write the header
            yaml_file.write(yaml_header)
            
            # Process each dataset
            for dataset_info in datasets:
                filename = dataset_info["file_name"]
                data = dataset_info["data"]
                
                logger.info(f"Processing dataset {filename} with {len(data)} entries")
                stats[filename] = {"total": len(data), "processed": 0, "skipped": 0}
                
                # Limit entries if specified
                entries_to_process = data
                if max_entries is not None and max_entries > 0 and len(data) > max_entries:
                    entries_to_process = data[:max_entries]
                    logger.info(f"Limiting to {len(entries_to_process)} entries from {filename}")
                
                # Create test entry for each item in dataset
                for i, entry in enumerate(entries_to_process):
                    # Extract input and output from dataset entry
                    if "input" not in entry or "output" not in entry:
                        logger.warning(f"Skipping entry {i} in {filename}: missing input or output")
                        stats[filename]["skipped"] += 1
                        skipped_entries += 1
                        continue
                    
                    # Get input (diff) and output (commit message)
                    diff_content = entry["input"]
                    commit_message = entry["output"]
                    
                    # No truncation - keep the entire diff content
                    
                    # Generate hash for diff content to create unique filename
                    diff_hash = generate_hash(diff_content)
                    diff_filename = f"diff_{diff_hash}.txt"
                    diff_path = diffs_dir / diff_filename
                    diff_relative_path = f"diffs/{diff_filename}"
                    
                    # Prepare complete content with system prompt + diff content
                    complete_content = f"{system_prompt_content}\n\n{diff_content}"
                    
                    # Check if we already created this file (avoid duplicates)
                    if diff_hash not in created_files:
                        # Write complete content to file
                        with open(diff_path, 'w', encoding='utf-8') as f:
                            f.write(complete_content)
                        created_files.add(diff_hash)
                        logger.debug(f"Created diff file: {diff_filename}")
                    
                    # Write test entry directly to YAML file with proper formatting
                    test_yaml = f"""- description: '{commit_message.replace("'", "''")}'
  vars:
    message: 'file://{diff_relative_path}'
    system_assertion_prompt: 'file://../assertion_prompts/diff_analyzer_assertion.md'
  assert:
  - type: llm-rubric
    value: '{{{{system_assertion_prompt}}}}'
"""
                    yaml_file.write(test_yaml)
                    
                    total_entries += 1
                    stats[filename]["processed"] += 1
                    
                    # Log progress every 10 entries
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(entries_to_process)} entries from {filename}")
        
        # Print summary
        logger.info(f"\nProcessing Summary:")
        logger.info(f"{'=' * 40}")
        for filename, file_stats in stats.items():
            logger.info(f"{filename}:")
            logger.info(f"  - Total entries: {file_stats['total']}")
            logger.info(f"  - Processed: {file_stats['processed']}")
            logger.info(f"  - Skipped: {file_stats['skipped']}")
        logger.info(f"{'=' * 40}")
        logger.info(f"Total tests generated: {total_entries}")
        logger.info(f"Total entries skipped: {skipped_entries}")
        logger.info(f"Total diff files created: {len(created_files)}")
        
        logger.info(f"Successfully created evaluation YAML with {total_entries} tests at {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating evaluation YAML: {e}")
        return False

def main():
    """
    Main function that orchestrates the evaluation set creation process.
    
    Parameters:
        None
        
    Returns:
        None
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Create evaluation sets from repository datasets.'
    )
    parser.add_argument('--datasets-dir', type=str, default='../../finetuning/repo_datasets',
                       help='Directory containing JSON datasets')
    parser.add_argument('--output-file', type=str, default='../eval_configs/diff_analyzer_eval_generated.yaml',
                       help='Output YAML file path')
    parser.add_argument('--max-entries', type=int, default=None,
                       help='Maximum entries to include per dataset (default: all entries)')
    parser.add_argument('--max-diff-size', type=int, default=100000,
                       help='Maximum size of diff content in characters (default: 100000)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    try:
        # Get script directory
        script_dir = Path(__file__).parent.resolve()
        
        # Set default paths relative to script directory if needed
        datasets_dir = args.datasets_dir
        if not os.path.isabs(datasets_dir):
            datasets_dir = script_dir / datasets_dir
        
        output_file = args.output_file
        if not os.path.isabs(output_file):
            output_file = script_dir / output_file
        
        # Print configuration
        logger.info(f"Starting evaluation set creation with configuration:")
        logger.info(f"  - Datasets directory: {datasets_dir}")
        logger.info(f"  - Output file: {output_file}")
        logger.info(f"  - Max entries per dataset: {args.max_entries if args.max_entries else 'All'}")
        logger.info(f"  - Max diff size: {args.max_diff_size} characters")
        
        # Load datasets
        datasets = load_json_datasets(datasets_dir)
        
        if not datasets:
            logger.error("No datasets loaded, exiting")
            return
        
        # Create evaluation YAML - pass script_dir to the function
        result = create_eval_yaml(datasets, output_file, script_dir, args.max_entries, args.max_diff_size)
        
        if result:
            logger.info("Evaluation set creation completed successfully")
            print(f"Evaluation set created successfully at {output_file}")
        else:
            logger.error("Evaluation set creation failed")
            print("Evaluation set creation failed. Check logs for details.")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 