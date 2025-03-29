"""
clean_dataset.py

Cleans a JSON dataset by removing entries that have outputs with more than 25 words.

Functions:
    count_words(text): Counts words in a text string
    clean_dataset(data, max_words): Removes entries with outputs exceeding word limit
    main(): Main function that processes the JSON file

Command Line Usage Examples:
    python clean_dataset.py input.json
    python clean_dataset.py input.json --output cleaned_output.json
    python clean_dataset.py input.json --max-words 15
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(funcName)s: %(message)s'
)
logger = logging.getLogger(__name__)

def count_words(text):
    """
    Count the number of words in a text string.
    
    Parameters:
        text (str): The text to count words in
        
    Returns:
        int: Number of words in the text
    """
    logger.debug(f"Counting words in text: {text[:50]}...")
    if not text or not isinstance(text, str):
        return 0
    
    # Split by whitespace and count non-empty parts
    words = [w for w in text.split() if w.strip()]
    return len(words)

def clean_dataset(data, max_words=25):
    """
    Remove entries with outputs exceeding the maximum word count.
    
    Parameters:
        data (list): List of dataset entries
        max_words (int): Maximum number of words allowed in output
        
    Returns:
        tuple: (Cleaned dataset, list of removed entries)
    """
    logger.debug(f"Cleaning dataset with {len(data)} entries | max_words: {max_words}")
    
    cleaned_data = []
    removed_entries = []
    
    for entry in data:
        if "output" not in entry:
            logger.warning(f"Entry missing 'output' field: {entry}")
            continue
            
        output = entry["output"]
        word_count = count_words(output)
        
        if word_count <= max_words:
            cleaned_data.append(entry)
        else:
            removed_entries.append({
                "entry": entry,
                "word_count": word_count
            })
    
    logger.info(f"Kept {len(cleaned_data)} entries, removed {len(removed_entries)} entries")
    return cleaned_data, removed_entries

def main():
    """
    Main function that processes the JSON file.
    
    Parameters:
        None
        
    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description='Clean a JSON dataset by removing entries with outputs exceeding a word limit.'
    )
    parser.add_argument('input_file', type=str, help='Input JSON file to clean')
    parser.add_argument('--output', type=str, help='Output file for cleaned dataset (default: input_file_cleaned.json)')
    parser.add_argument('--max-words', type=int, default=25, help='Maximum number of words allowed in output (default: 25)')
    parser.add_argument('--log-removed', action='store_true', help='Write removed entries to a separate JSON file')
    
    try:
        args = parser.parse_args()
        
        input_file = args.input_file
        max_words = args.max_words
        
        # Set default output file if not provided
        if args.output:
            output_file = args.output
        else:
            input_path = Path(input_file)
            output_file = str(input_path.with_name(f"{input_path.stem}_cleaned{input_path.suffix}"))
        
        # Load input JSON file
        logger.info(f"Loading dataset from {input_file}")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                logger.error(f"Input file does not contain a JSON array/list")
                sys.exit(1)
                
            logger.info(f"Loaded {len(data)} entries from input file")
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in input file: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error reading input file: {e}")
            sys.exit(1)
        
        # Clean dataset
        cleaned_data, removed_entries = clean_dataset(data, max_words)
        
        # Save cleaned dataset
        logger.info(f"Saving cleaned dataset to {output_file}")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=2)
            
            logger.info(f"Saved {len(cleaned_data)} entries to {output_file}")
            
        except Exception as e:
            logger.error(f"Error writing output file: {e}")
            sys.exit(1)
        
        # Log removed entries if requested
        if args.log_removed and removed_entries:
            removed_file = str(Path(output_file).with_name(f"{Path(output_file).stem}_removed{Path(output_file).suffix}"))
            logger.info(f"Saving {len(removed_entries)} removed entries to {removed_file}")
            
            try:
                with open(removed_file, 'w', encoding='utf-8') as f:
                    json.dump(removed_entries, f, indent=2)
            except Exception as e:
                logger.error(f"Error writing removed entries file: {e}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"  Original entries: {len(data)}")
        print(f"  Cleaned entries: {len(cleaned_data)}")
        print(f"  Removed entries: {len(removed_entries)}")
        print(f"  Cleaned dataset saved to: {output_file}")
        if args.log_removed and removed_entries:
            print(f"  Removed entries saved to: {removed_file}")
        
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
        print("\nProgram terminated by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 