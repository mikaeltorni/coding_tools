# GitHub Repository Scraper

A Python tool for scraping GitHub repositories including all branches and analyzing Git diffs using Gemma 3 model.

## Features

- Clone a GitHub repository locally
- Fetch and process all branches from the repository
- Analyze Git diffs using Gemma 3 model
- Generate dataset in Alpaca format

## Requirements

- Python 3.8+
- Git
- Llama server running with Gemma 3 model

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/github-repo-scraper.git
cd github-repo-scraper
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Make sure you have a Llama server running with the Gemma 3 model.

## Usage

```bash
python github_repository_scraper.py <repo_url> [options]
```

### Command-line Arguments

- `repo_url`: URL of the GitHub repository to scrape (required)
- `--target-dir`: Target directory to clone the repository into (default: ./cloned_repos)
- `--server-url`: URL of the Llama server (default: http://localhost:8080)
- `--output-file`: Output file for the Alpaca dataset (default: github_diff_dataset.json)
- `--temperature`: Temperature parameter for text generation (default: 0.0)
- `--max-tokens`: Maximum number of tokens to generate (default: 1024)

### Example

```bash
python github_repository_scraper.py https://github.com/sadmann7/shadcn-table.git --server-url http://localhost:8080
```

## Output

The script generates a JSON file in Alpaca dataset format with the following structure:

```json
[
  {
    "instruction": "Read the Git diff and make a short, 10-15 word summary with one of the following tags: feat, fix, docs, style, refactor, perf, test, build, ci, chore",
    "input": "<git diff content>",
    "output": "<model classification>",
    "text": "<formatted instruction, input, and output>"
  },
  ...
]
```

## How It Works

1. The script clones the specified GitHub repository to a local directory
2. It fetches all branches from the remote repository
3. For each branch, it checks out the branch and gets the diff compared to the main branch
4. The diff is sent to the Gemma 3 model for analysis
5. The model classifies the changes with one of the conventional commit tags
6. The results are saved as a dataset in Alpaca format