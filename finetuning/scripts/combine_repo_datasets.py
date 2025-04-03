"""
combine_repo_datasets.py

Combines multiple JSON dataset files from repo_datasets directory into a single consolidated dataset
and optionally converts it to Parquet format.

Functions:
    combine_datasets(source_dir, output_file, convert_to_parquet): Finds and combines JSON files into a single dataset
    find_json_files(directory): Finds all JSON files in the specified directory
    load_json_file(file_path): Loads a JSON file and returns its contents
    save_combined_dataset(data, output_file): Saves the combined dataset to a JSON file
    extract_commit_prefix(commit_message): Extracts the commit type prefix from a commit message
    print_prefix_stats(prefix_counts, total_entries): Prints statistics about commit prefix frequencies

Command Line Usage Examples:
    python combine_repo_datasets.py
    python combine_repo_datasets.py --source-dir ../repo_datasets --output-file ../combined_full_dataset.json
    python combine_repo_datasets.py --no-parquet  # Skip converting to Parquet format
"""
import argparse
import json
import logging
import os
import sys
import re
from pathlib import Path
from collections import Counter

# Import the JSON to Parquet conversion function
from json_to_parquet import convert_json_to_parquet

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s:%(funcName)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Define conventional commit types (same as in github_repository_scraper.py)
CONVENTIONAL_TYPES = ['feat', 'fix', 'docs', 'style', 'refactor', 'perf', 'test', 'build', 'ci', 'chore', 'grunt', 'dist']

def extract_commit_prefix(commit_message):
    """
    Extract the prefix (type and scope) from a commit message.
    
    Parameters:
        commit_message (str): The commit message to analyze
        
    Returns:
        str: The extracted prefix or None if not a conventional commit
    """
    if not commit_message:
        return None
        
    # Get the first part before any colon
    parts = commit_message.strip().split(':', 1)
    if len(parts) < 2:
        return None
    
    prefix = parts[0].lower()
    
    # Extract just the type from the prefix (without scope)
    match = re.match(r'^([\w-]+)(?:\([\w-]+\))?$', prefix)
    if match and match.group(1) in CONVENTIONAL_TYPES:
        return prefix
    
    return None

def print_prefix_stats(prefix_counts, total_entries):
    """
    Print statistics about commit prefix frequencies in the dataset.
    
    Parameters:
        prefix_counts (Counter): Counter of prefix frequencies
        total_entries (int): Total number of entries in the dataset
        
    Returns:
        None
    """
    print("\n===== COMMIT PREFIX STATISTICS =====")
    print(f"Total entries in dataset: {total_entries}")
    
    # Extract just the commit types (without scopes)
    type_counts = Counter()
    for prefix, count in prefix_counts.items():
        commit_type = prefix.split('(')[0] if '(' in prefix else prefix
        type_counts[commit_type] += count
    
    # Print summary of commit types
    print("\nCommit types summary:")
    for commit_type, count in type_counts.most_common():
        percentage = (count / total_entries) * 100 if total_entries > 0 else 0
        print(f"  {commit_type}: {count} ({percentage:.1f}%)")
    
    # Print the top prefixes with scopes
    print("\nTop 20 prefixes with scopes:")
    for prefix, count in prefix_counts.most_common(20):
        percentage = (count / total_entries) * 100 if total_entries > 0 else 0
        print(f"  {prefix}: {count} ({percentage:.1f}%)")
    
    # Print entries without recognized prefixes
    non_conventional = total_entries - sum(prefix_counts.values())
    if non_conventional > 0:
        percentage = (non_conventional / total_entries) * 100 if total_entries > 0 else 0
        print(f"\nEntries without recognized prefixes: {non_conventional} ({percentage:.1f}%)")
    
    print("====================================")

def find_json_files(directory):
    """
    Find all JSON files in the specified directory.
    
    Parameters:
        directory (str or Path): Directory to search for JSON files
        
    Returns:
        list: List of Path objects for all JSON files found
    """
    logger.debug(f"Searching for JSON files in: {directory}")
    directory_path = Path(directory)
    
    if not directory_path.exists():
        logger.error(f"Directory not found: {directory}")
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory_path.is_dir():
        logger.error(f"Not a directory: {directory}")
        raise NotADirectoryError(f"Not a directory: {directory}")
    
    json_files = list(directory_path.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files")
    
    return json_files

def load_json_file(file_path):
    """
    Load a JSON file and return its contents.
    
    Parameters:
        file_path (str or Path): Path to the JSON file
        
    Returns:
        list: Contents of the JSON file
    """
    path = Path(file_path)
    logger.debug(f"Loading file: {path}")
    
    try:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logger.warning(f"File {path} does not contain a JSON array, skipping")
            return []
        
        logger.info(f"Loaded {len(data)} entries from {path}")
        return data
    
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file {path}: {e}")
        print(f"Error: Could not parse JSON file {path}: {e}", file=sys.stderr)
        return []
    
    except Exception as e:
        logger.error(f"Error loading file {path}: {e}")
        print(f"Error: Could not load file {path}: {e}", file=sys.stderr)
        return []

def save_combined_dataset(data, output_file):
    """
    Save the combined dataset to a JSON file.
    
    Parameters:
        data (list): The combined dataset
        output_file (str or Path): Path to the output file
        
    Returns:
        bool: True if successful, False otherwise
    """
    output_path = Path(output_file)
    logger.debug(f"Saving combined dataset to: {output_path}")
    
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Successfully saved {len(data)} entries to {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving combined dataset: {e}")
        print(f"Error: Could not save combined dataset: {e}", file=sys.stderr)
        return False

def combine_datasets(source_dir, output_file, convert_to_parquet=True):
    """
    Find and combine JSON files from the source directory into a single dataset.
    
    Parameters:
        source_dir (str or Path): Directory containing JSON files to combine
        output_file (str or Path): Path to save the combined dataset
        convert_to_parquet (bool): Whether to also convert the dataset to Parquet format
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.debug(f"source_dir: {source_dir} | output_file: {output_file} | convert_to_parquet: {convert_to_parquet}")
    
    # Find all JSON files in the source directory
    try:
        json_files = find_json_files(source_dir)
        
        if not json_files:
            logger.warning(f"No JSON files found in {source_dir}")
            print(f"Warning: No JSON files found in {source_dir}")
            return False
        
        # Combine all datasets
        combined_data = []
        unique_entries = set()  # Track entries we've already seen to avoid duplicates
        prefix_counts = Counter()  # Count prefix frequencies
        
        for file_path in json_files:
            logger.info(f"Processing file: {file_path}")
            
            # Load the dataset
            dataset = load_json_file(file_path)
            
            # Add entries to the combined dataset, avoiding duplicates
            for entry in dataset:
                # Use a tuple of instruction, input, output as a hash for deduplication
                try:
                    entry_hash = (
                        entry.get('instruction', ''), 
                        entry.get('input', ''),
                        entry.get('output', '')
                    )
                    
                    if entry_hash not in unique_entries:
                        combined_data.append(entry)
                        unique_entries.add(entry_hash)
                        
                        # Extract and count prefix from output
                        output = entry.get('output', '')
                        prefix = extract_commit_prefix(output)
                        if prefix:
                            prefix_counts[prefix] += 1
                except (TypeError, AttributeError):
                    # Skip entries that aren't dictionaries or don't have the right fields
                    logger.warning(f"Skipping invalid entry: {entry}")
                    continue
            
            logger.info(f"Combined dataset now has {len(combined_data)} entries")
        
        # Save the combined dataset
        if combined_data:
            # Save to JSON
            success = save_combined_dataset(combined_data, output_file)
            
            if not success:
                print(f"Failed to save combined dataset to {output_file}")
                return False
            
            print(f"Successfully combined {len(json_files)} files into {output_file}")
            print(f"Total entries: {len(combined_data)}")
            
            # Print prefix statistics
            print_prefix_stats(prefix_counts, len(combined_data))
            
            # Convert to Parquet if requested
            if convert_to_parquet:
                parquet_path = str(Path(output_file).with_suffix('.parquet'))
                try:
                    logger.info(f"Converting combined dataset to Parquet format: {parquet_path}")
                    convert_json_to_parquet(output_file, parquet_path)
                    print(f"Successfully converted combined dataset to Parquet format: {parquet_path}")
                except Exception as e:
                    logger.error(f"Error converting to Parquet: {e}")
                    print(f"Warning: Failed to convert to Parquet format: {e}", file=sys.stderr)
                    # Don't consider this a failure of the main operation
            
            return True
        else:
            logger.warning("No valid entries found in any of the JSON files")
            print("Warning: No valid entries found in any of the JSON files")
            return False
        
    except Exception as e:
        logger.error(f"Error combining datasets: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return False

def main():
    """
    Parse command line arguments and combine datasets.
    
    Parameters:
        None
        
    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description='Combine multiple JSON dataset files into a single dataset and convert to Parquet'
    )
    parser.add_argument('--source-dir', type=str, default='../repo_datasets',
                       help='Directory containing JSON files to combine (default: ../repo_datasets)')
    parser.add_argument('--output-file', type=str, default='../combined_full_dataset.json',
                       help='Path to save the combined dataset (default: ../combined_full_dataset.json)')
    parser.add_argument('--no-parquet', action='store_true',
                        help='Skip converting the combined dataset to Parquet format')
    
    args = parser.parse_args()
    
    # Combine the datasets
    success = combine_datasets(args.source_dir, args.output_file, not args.no_parquet)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 