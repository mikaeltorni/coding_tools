"""
temp_diff_cleaner.py

Cleans Git diff data in JSON files by removing commit metadata, keeping only the actual diff.

Functions:
    clean_diff_data(input_json, output_json): Cleans the diff data in the input JSON file and saves to output file.
    clean_input_field(input_text): Removes commit metadata from the input field.

Command Line Usage Examples:
    python temp_diff_cleaner.py input.json output.json
    python temp_diff_cleaner.py github_diff_dataset.json github_diff_dataset_clean.json
"""
import json
import logging
import argparse
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s:%(funcName)s: %(message)s'
)
logger = logging.getLogger(__name__)

def clean_input_field(input_text):
    """
    Removes commit metadata from the input field, keeping only the actual diff.
    
    Parameters:
        input_text (str): Original input text containing commit metadata and diff
        
    Returns:
        str: Cleaned input text with only the diff part
    """
    logger.debug(f"Input text length: {len(input_text)}")
    
    # Find where the diff starts
    diff_start_idx = input_text.find("diff --git")
    
    if diff_start_idx != -1:
        # Keep only the diff part
        cleaned_text = input_text[diff_start_idx:]
        logger.debug(f"Cleaned text length: {len(cleaned_text)}")
        return cleaned_text
    else:
        # If "diff --git" not found, return the original text
        logger.warning("No 'diff --git' found in input text, returning original")
        return input_text

def clean_diff_data(input_json, output_json):
    """
    Cleans the diff data in the input JSON file and saves to output file.
    
    Parameters:
        input_json (str): Path to input JSON file
        output_json (str): Path to output JSON file
        
    Returns:
        None
    """
    logger.debug(f"input_json: {input_json} | output_json: {output_json}")
    
    try:
        # Check if input file exists
        input_path = Path(input_json)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_json}")
            raise FileNotFoundError(f"Input file not found: {input_json}")
            
        # Load JSON data
        logger.info(f"Loading JSON data from {input_json}")
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        logger.info(f"Loaded {len(data)} entries")
        
        # Process each entry
        cleaned_entries = 0
        for entry in data:
            if "input" in entry:
                # Clean the input field
                original_input = entry["input"]
                cleaned_input = clean_input_field(original_input)
                
                # Update entry if cleaned
                if cleaned_input != original_input:
                    entry["input"] = cleaned_input
                    cleaned_entries += 1
        
        logger.info(f"Cleaned {cleaned_entries} entries")
        
        # Save to output file
        logger.info(f"Saving cleaned data to {output_json}")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved {len(data)} entries to {output_json}")
        print(f"Successfully cleaned {cleaned_entries} entries out of {len(data)} total entries.")
        print(f"Saved to {output_json}")
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        print(f"Error: Invalid JSON in {input_json}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error cleaning diff data: {e}")
        print(f"Error: {e}")
        sys.exit(1)

def main():
    """
    Main function to parse command line arguments and run the cleaner.
    
    Parameters:
        None
        
    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description='Clean Git diff data in JSON files by removing commit metadata.'
    )
    parser.add_argument('input_json', help='Input JSON file path')
    parser.add_argument('output_json', nargs='?', help='Output JSON file path (defaults to overwriting input file)')
    
    args = parser.parse_args()
    
    input_json = args.input_json
    output_json = args.output_json or input_json
    
    clean_diff_data(input_json, output_json)

if __name__ == "__main__":
    main() 