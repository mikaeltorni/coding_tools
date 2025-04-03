# Repository Dataset Evaluation Generator

This directory contains scripts for creating evaluation sets from repository datasets.

## create_eval_set_from_repo_datasets.py

This script generates evaluation YAML files for the Diff Analyzer model by processing JSON datasets containing Git diffs and commit messages.

### Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

```bash
python create_eval_set_from_repo_datasets.py [OPTIONS]
```

#### Options

- `--datasets-dir PATH`: Directory containing JSON datasets (default: ../../finetuning/repo_datasets)
- `--output-file PATH`: Output YAML file path (default: ../eval_configs/diff_analyzer_eval_generated.yaml)
- `--max-entries N`: Maximum entries to include per dataset (default: all entries)
- `--max-diff-size N`: Maximum size of diff content in characters (default: 100000)
- `--verbose`, `-v`: Enable verbose logging for debugging

### Example

```bash
# Generate evaluation set with default settings (process all entries)
python create_eval_set_from_repo_datasets.py

# Generate evaluation set with custom path and entry limit
python create_eval_set_from_repo_datasets.py --datasets-dir /path/to/datasets --max-entries 20

# Generate evaluation set with smaller diff size limit
python create_eval_set_from_repo_datasets.py --max-diff-size 50000 --verbose
```

### Output

The script will:

1. Read all JSON files in the specified datasets directory
2. Create a "diffs" directory to store individual diff files
3. Write each diff to a separate text file with an MD5 hash-based filename
4. Create YAML entries that reference the diff files using the file:// protocol
5. Output status messages and a summary of processed entries

#### File Structure

The script generates the following file structure:

```
eval_configs/
├── diff_analyzer_eval_generated.yaml   # Main evaluation YAML
└── diffs/                             # Directory for diff files
    ├── diff_<hash1>.txt               # Individual diff files
    ├── diff_<hash2>.txt
    └── ...
```

#### Summary Report

After processing, the script will generate a summary report showing:
- How many entries were processed from each dataset file
- How many entries were skipped (due to missing input/output fields)
- Total number of tests generated
- Total number of unique diff files created

The script expects the assertion prompt file to exist at the path `assertion_prompts/diff_analyzer_assertion.md`. The generated YAML can be used with the evaluation framework to test the Diff Analyzer model's performance. 