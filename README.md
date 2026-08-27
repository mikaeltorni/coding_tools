# coding_tools — Local AI Git Commit Message Generator

[![Last commit](https://img.shields.io/github/last-commit/mikaeltorni/coding_tools)](https://github.com/mikaeltorni/coding_tools/commits/master)
[![Commit activity](https://img.shields.io/github/commit-activity/m/mikaeltorni/coding_tools)](https://github.com/mikaeltorni/coding_tools/graphs/commit-activity)
[![Issues](https://img.shields.io/github/issues/mikaeltorni/coding_tools)](https://github.com/mikaeltorni/coding_tools/issues)

coding_tools is a local AI git commit message generator that writes conventional commits from diffs for developers.

It uses a fine-tuned Gemma 3 (1B) language model running locally through
[llama.cpp](https://github.com/ggml-org/llama.cpp) with CUDA/GPU acceleration.
Monitor one or many Git repositories and request an AI-written commit message
with a hotkey; no API key is required.

**Topics:** git · commit-message-generator · conventional-commits · gemma ·
fine-tuning · llama-cpp · gguf · local-llm · cuda · ai-developer-tools · python

## Contents

- [Local AI Git Commit Message Generator Features](#local-ai-git-commit-message-generator-features)
- [Installation](#installation)
- [Local AI Git Commit Message Generator Usage Examples](#local-ai-git-commit-message-generator-usage-examples)
- [Configuration](#configuration)
- [Troubleshooting and FAQ](#troubleshooting-and-faq)
- [Contributing](#contributing)

## Quickstart

After installing Conda and the local llama.cpp server, inspect the CLI options:

```bash
conda create --name ct python=3.12.8 -y
conda activate ct
python3 main.py --help
```

The program accepts one or more Git repository paths and sends their diffs to
the local server when the configured hotkey is pressed.

### AI System Information
- **Purpose**: Automated generation of git commit messages based on code diff analysis
- **AI Model**: Fine-tuned Gemma 3 (1B parameters) specialized for code change classification

---

## Local AI Git Commit Message Generator Overview

This tool uses a fine-tuned Gemma 3 model to automatically analyze git diffs and generate appropriate commit messages following conventional commit standards. The system monitors git repositories in real-time and provides AI-generated commit message suggestions when triggered.

For additional standalone automation utilities, see the related
[`scripts`](https://github.com/mikaeltorni/scripts) collection.

## Local AI Git Commit Message Generator Features
- Real-time git repository monitoring
- AI-powered diff analysis and commit message generation
- Multi-repository support
- Conventional commit format compliance (feat:, fix:, docs:, etc.)

## How the Local AI Git Commit Message Generator Works
1. Monitors specified git repositories for changes
2. When triggered (via hotkey), analyzes current git diff using AI
3. Generates appropriate commit message based on code changes

---

## Installation

This guide explains how to set up your environment to run optimized inference with llama.cpp. Follow these steps:

---

### 1. Create and Activate the Conda Environment

Make sure you have [Conda](https://www.anaconda.com/docs/getting-started/miniconda/main) installed (via Anaconda or Miniconda). Create a new environment named **ct** with Python version 3.12.8:

```bash
conda create --name ct python=3.12.8
conda activate ct
```

---

### 2. GPU Acceleration Setup for Llama.cpp

### Prerequisites
- Install CUDA Toolkit 12.8 (or newer) from the official NVIDIA website.
- Verify that your NVIDIA GPU (for example, an RTX 4090 with compute capability 8.9 [Get the value from here, remove dot and insert it to the -DCMAKE_CUDA_ARCHITECTURES parameter](https://developer.nvidia.com/cuda-gpus)) is supported.

### Build Instructions
1. Open a Windows Command Prompt.
2. Configure and build llama.cpp by running the following commands:
```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES="89" && cmake --build build --config Release
```

---

### 3. Download a GGUF Model

1. **Download the Gemma 3 1B Model:**
   - Visit [unsloth/gemma-3-1b-it-GGUF](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/tree/main)
   - Download the Q4_K_M quantized model file (ending with .gguf extension)
   - This quantization level provides a good balance between model size and inference quality

---

### 4. Running the Llama Server

1. Open a new Command Prompt.
2. Set the CUDA_VISIBLE_DEVICES environment variable and start the server with your model by running:
```bash
set CUDA_VISIBLE_DEVICES=-0 && ..\llama.cpp\build\bin\Release\llama-server --model ..\models\gemma-3-1b-it-Q4_K_M.gguf --n-gpu-layers 420
```
(set up a high gpu layers number to increase the models inference speed)

3. The server will start and display information about the model and inference settings.
4. Connect to the server using HTTP requests to localhost on the default port.

---

### Additional Configuration Parameters

- Other useful parameters:
  - `--ctx-size`: Context window size (default: 2048)
  - `--batch-size`: Batch size for prompt processing
  - `--threads`: Number of CPU threads to use
  - `--stream`: Enable streaming mode

---

## Local AI Git Commit Message Generator Usage Examples

You can monitor a single Git repository:
```bash
python main.py /path/to/repo
```

### Multi-Repository Support

The program now supports monitoring multiple Git repositories simultaneously:
```bash
python main.py /path/to/repo1 /path/to/repo2 /path/to/repo3
```

When monitoring multiple repositories:
- The program processes each repository individually when you press the hotkey
- Diff content for each repository is saved to separate files (output_repo-name.txt)
- LLM responses and commits will be labeled with the repository name
- Invalid repositories will be skipped with a warning

### Other Options

Additional command-line options:
```bash
python main.py /path/to/repo [options]

Options:
  --server-url URL        URL of the llama server (default: http://localhost:8080)
  --hotkey KEY            Hotkey combination to trigger LLM feedback (default: alt+q)
  --temperature TEMP      Temperature parameter for text generation (default: 0.7)
  --max-tokens TOKENS     Maximum number of tokens to generate (default: 512)
  --context-length LENGTH Context length for the model (default: 2048)
```

## Configuration

The default server URL, hotkey, temperature, maximum output tokens, and context
length are defined in `data/model_config.py` and can be overridden with the
matching command-line options. Repository paths are positional arguments, and
multiple paths enable multi-repository monitoring.

## Troubleshooting and FAQ

### What does coding_tools generate?

It proposes conventional Git commit messages from the current diff using a
locally hosted Gemma model. It does not commit changes automatically unless the
configured keyboard action requests that behavior.

### Does the tool send diffs to a cloud API?

No. The documented setup points to a local llama.cpp server and does not require
an API key. Keep the model server and monitored repositories on the same trusted
machine.

### Can I monitor more than one repository?

Yes. Pass multiple repository paths after `main.py`; each valid repository is
validated and monitored independently, with labeled outputs for each path.

### Why does the program need CUDA?

CUDA is the documented acceleration path for the fine-tuned GGUF model. A CPU
build of llama.cpp may work if configured separately, but this repository does
not claim equivalent performance or provide a CPU setup guide.

### Which server URL does the program use?

The default is `http://localhost:8080`, matching the README's llama-server
example. Use `--server-url URL` when the local server listens elsewhere.

## Contributing

Keep the local/offline behavior explicit, document new command-line flags, and
test single- and multi-repository argument handling before opening a pull
request. Never commit model files, credentials, or generated output.

## License

Released under the [MIT License](LICENSE.md).
