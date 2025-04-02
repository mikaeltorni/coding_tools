"""
create_eval_set_from_repo_datasets.py

Script to create evaluation sets from repository datasets in JSON format.
This script loops through JSON files in the repo_datasets folder and generates
evaluation entries for each item in the datasets.

Functions:
    load_json_datasets(directory): Loads all JSON datasets from a directory
    create_eval_yaml(datasets, output_path): Creates evaluation YAML file from datasets
    main(): Main function that orchestrates the evaluation set creation process

Command Line Usage Examples:
    python create_eval_set_from_repo_datasets.py
    python create_eval_set_from_repo_datasets.py --datasets-dir ../../finetuning/repo_datasets
    python create_eval_set_from_repo_datasets.py --output-file diff_analyzer_eval_generated.yaml
"""
import os
import json
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
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
                    datasets.append({
                        "file_name": json_file.name,
                        "data": data
                    })
                logger.info(f"Successfully loaded {len(data)} entries from {json_file.name}")
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
                continue
                
        return datasets
        
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise

def create_eval_yaml(datasets, output_path, max_entries=None):
    """
    Creates evaluation YAML file from datasets.
    
    Parameters:
        datasets (list): List of datasets loaded from JSON files
        output_path (str): Path to output the YAML file
        max_entries (int, optional): Maximum number of entries to include per dataset
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.debug(f"output_path: {output_path} | max_entries: {max_entries}")
    
    try:
        # Create base YAML structure
        yaml_structure = {
            "description": "Diff Analyzer Agent Evals - Auto-generated from repository datasets",
            "prompts": [
                "file://formats/default.json"
            ],
            "providers": [
                {
                    "id": "llama:gemma-3-1b-it-Q4_K_M",
                    "config": {
                        "temperature": 0,
                        "max_tokens": 4096,
                        "top_p": 0.9,
                        "apiEndpoint": "${LLAMA_BASE_URL:-http://localhost:8080}"
                    }
                }
            ],
            "defaultTest": {
                "options": {
                    "provider": "openai:gpt-4o-mini-2024-07-18"
                }
            },
            "tests": []
        }
        
        # Check if assertion prompt file exists
        assertion_file = Path(output_path).parent / "assertion_prompts" / "diff_analyzer_assertion.md"
        
        if not assertion_file.exists():
            logger.error(f"Assertion prompt file not found at {assertion_file}")
            raise FileNotFoundError(f"Assertion prompt file not found at {assertion_file}")
        
        logger.info(f"Using existing assertion prompt at {assertion_file}")
        
        # Counter for test entries
        total_entries = 0
        
        # Process each dataset
        for dataset_info in datasets:
            filename = dataset_info["file_name"]
            data = dataset_info["data"]
            
            logger.info(f"Processing dataset {filename} with {len(data)} entries")
            
            # Limit entries if specified
            entries_to_process = data
            if max_entries is not None and max_entries > 0:
                entries_to_process = data[:max_entries]
                logger.info(f"Limiting to {len(entries_to_process)} entries from {filename}")
            
            # Create test entry for each item in dataset
            for i, entry in enumerate(entries_to_process):
                # Extract input and output from dataset entry
                if "input" not in entry or "output" not in entry:
                    logger.warning(f"Skipping entry {i} in {filename}: missing input or output")
                    continue
                
                # Get input (diff) and output (commit message)
                diff_content = entry["input"]
                commit_message = entry["output"]
                
                # Create test entry
                test_entry = {
                    "description": commit_message,
                    "vars": {
                        "system_prompt": "file://../data/prompts/system/diff_analyzer.txt",
                        "user_prompt": diff_content,
                        "system_assertion_prompt": "file://assertion_prompts/diff_analyzer_assertion.md"
                    },
                    "assert": [
                        {
                            "type": "llm-rubric",
                            "value": "{{system_assertion_prompt}}"
                        }
                    ]
                }
                
                # Add test entry to YAML structure
                yaml_structure["tests"].append(test_entry)
                total_entries += 1
                
                # Log progress every 10 entries
                if i % 10 == 0:
                    logger.info(f"Processed {i} entries from {filename}")
        
        # Write YAML file
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_structure, f, default_flow_style=False, sort_keys=False)
        
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
    parser.add_argument('--output-file', type=str, default='diff_analyzer_eval_generated.yaml',
                       help='Output YAML file path')
    parser.add_argument('--max-entries', type=int, default=10,
                       help='Maximum entries to include per dataset (default: 10)')
    
    args = parser.parse_args()
    
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
        
        logger.info(f"Starting evaluation set creation")
        logger.info(f"Datasets directory: {datasets_dir}")
        logger.info(f"Output file: {output_file}")
        
        # Load datasets
        datasets = load_json_datasets(datasets_dir)
        
        if not datasets:
            logger.error("No datasets loaded, exiting")
            return
        
        # Create evaluation YAML
        result = create_eval_yaml(datasets, output_file, args.max_entries)
        
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