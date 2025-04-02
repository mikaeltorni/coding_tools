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
- `--output-file PATH`: Output YAML file path (default: diff_analyzer_eval_generated.yaml)
- `--max-entries N`: Maximum entries to include per dataset (default: 10)

### Example

```bash
# Generate evaluation set with default settings
python create_eval_set_from_repo_datasets.py

# Generate evaluation set with custom path and entry limit
python create_eval_set_from_repo_datasets.py --datasets-dir /path/to/datasets --max-entries 20
```

### Output

The script will:

1. Read all JSON files in the specified datasets directory
2. Use the existing assertion prompt from `assertion_prompts/diff_analyzer_assertion.md`
3. Generate a YAML file containing evaluation tests for each dataset entry
4. Output status messages with information about the process

The script expects the assertion prompt file to exist at the path `assertion_prompts/diff_analyzer_assertion.md`. The generated YAML can be used with the evaluation framework to test the Diff Analyzer model's performance. 