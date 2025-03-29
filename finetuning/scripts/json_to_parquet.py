"""
json_to_parquet.py

Converts JSON file to Parquet format for more efficient data processing.

Functions:
    convert_json_to_parquet(json_path, parquet_path): Converts a JSON file to Parquet format.

Command Line Usage Examples:
    python json_to_parquet.py
    python json_to_parquet.py --input github_diff_dataset.json --output github_diff_dataset.parquet
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# External packages
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s:%(funcName)s: %(message)s'
)

logger = logging.getLogger(__name__)

def convert_json_to_parquet(json_path, parquet_path):
    """
    Convert JSON file to Parquet format.
    
    Parameters:
        json_path (str): Path to the input JSON file
        parquet_path (str): Path to save the output Parquet file
        
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    logger.debug(f"json_path: {json_path} | parquet_path: {parquet_path}")
    
    # Validate input file exists
    input_path = Path(json_path)
    if not input_path.exists():
        logger.error(f"Input file not found: {json_path}")
        raise FileNotFoundError(f"Input file not found: {json_path}")
    
    if input_path.stat().st_size == 0:
        logger.error(f"Input file is empty: {json_path}")
        raise ValueError(f"Input file is empty: {json_path}")
    
    try:
        # Check if the file is in JSON Lines format based on extension
        is_jsonl = input_path.suffix.lower() in ['.jsonl', '.jl']
        lines_param = is_jsonl
        
        logger.info(f"Reading JSON file: {json_path} (lines format: {lines_param})")
        df = pd.read_json(json_path, lines=lines_param)
        
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Saving to Parquet file: {parquet_path}")
        df.to_parquet(parquet_path, index=False)
        
        logger.info("Conversion completed successfully")
        return True
        
    except ValueError as e:
        logger.error(f"JSON parsing error: {e}")
        
        # Try the alternative format if first attempt fails
        try:
            logger.info(f"Retrying with lines={not lines_param}")
            df = pd.read_json(json_path, lines=not lines_param)
            logger.info(f"DataFrame shape: {df.shape}")
            df.to_parquet(parquet_path, index=False)
            logger.info("Conversion completed successfully on retry")
            return True
        except Exception as retry_e:
            logger.error(f"Retry also failed: {retry_e}")
            raise ValueError(f"Could not parse JSON file in either format: {e}") from retry_e
            
    except Exception as e:
        logger.error(f"Unexpected error during conversion: {e}")
        raise RuntimeError(f"Failed to convert JSON to Parquet: {e}") from e

def main():
    """
    Parse command line arguments and execute the conversion.
    
    Parameters:
        None
        
    Returns:
        None
    """
    parser = argparse.ArgumentParser(description='Convert JSON file to Parquet format')
    parser.add_argument('--input', '-i', type=str, default='github_diff_dataset.json',
                        help='Input JSON file path')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output Parquet file path (defaults to input filename with .parquet extension)')
    
    args = parser.parse_args()
    
    # If output is not specified, use input name with .parquet extension
    output_path = args.output if args.output else Path(args.input).stem + '.parquet'
    
    try:
        convert_json_to_parquet(args.input, output_path)
        print(f"Successfully converted {args.input} to {output_path}")
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()