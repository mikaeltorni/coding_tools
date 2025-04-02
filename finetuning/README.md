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
    "instruction": "You are an expert at analyzing Git diffs and classifying their changes in short, 10-15 word summaries. Make sure to read the diffs line-by-line for the provided diff by reading what has been added, and removed on the currently unstaged files in the repository. Then proceed to classify it with one of the tags, that are the following: feat: A new feature, fix: A bug fix, docs: Documentation only changes, style: Changes that do not affect the meaning of the code, refactor: A code change that neither fixes a bug nor adds a feature, perf: A code change that improves performance, test: Adding missing tests or correcting existing tests, build: Changes that affect the build system or external dependencies, ci: Changes to CI configuration files and scripts, chore: Other changes that don't modify src or test files. You can also use these tags with scopes in parentheses to provide more context, for example: fix(deps): Update dependency versions, feat(auth): Add new authentication method. Your response should be a short 10-15 word summary starting with the tag. For example: 'feat: implemented user authentication with JWT tokens' or 'fix(deps): updated npm dependencies to fix security vulnerabilities'. By any means, do not exceed the 15 word limit, and do not produce anything more than this one sentence.",
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