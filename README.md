# Model Inference with llama.cpp

This guide explains how to set up your environment to run optimized inference with llama.cpp. Follow these steps:

---

## 1. Create and Activate the Conda Environment

Make sure you have [Conda](https://www.anaconda.com/docs/getting-started/miniconda/main) installed (via Anaconda or Miniconda). Create a new environment named **ct** with Python version 3.13.1:

```bash
conda create --name ct python=3.13.1
conda activate ct
```

---

## 2. Install Dependencies

With the Conda environment activated, install the dependencies using pip:

```bash
pip install -r requirements.txt
```

For GPU acceleration (recommended), reinstall llama-cpp-python with CUDA support:

```bash
pip uninstall -y llama-cpp-python
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

---

## 3. Download a GGUF Model

1. **Download a Compatible GGUF Model:**
   - Visit [Hugging Face](https://huggingface.co/) and search for models with GGUF format
   - Common GGUF models include llama, gemma, mistral, and others with different quantization levels
   - Download the model file (typically ending with .gguf extension)
   - Recommended models for good performance:
     - [TheBloke/Llama-3-8B-Instruct-GGUF](https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF)
     - [TheBloke/Mistral-7B-Instruct-v0.2-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
     - [TheBloke/Gemma-3-1b-it-GGUF](https://huggingface.co/TheBloke/Gemma-3-1b-it-GGUF)

2. **Choose the Right Quantization Level:**
   - Q4_K_M is typically a good balance between performance and quality
   - Q2_K offers smaller size but lower quality
   - Q8_0 offers higher quality but larger size

---

## 4. Using the Model in Your Code

```python
from src.models import ModelManager

# Initialize the model with optimized settings
model_path = "path/to/your/model.gguf"
model, config = ModelManager.load_model(model_path)

# Example of generating text
output = model.create_completion(
    "Your prompt text here",
    max_tokens=config["max_tokens"],
    temperature=config["temperature"],
    top_p=config["top_p"],
    top_k=config["top_k"],
    repeat_penalty=config["repeat_penalty"]
)

# Get the generated text
generated_text = output["choices"][0]["text"]
```

---

## Performance Optimization

The ModelManager is already configured with optimal default settings, but you can fine-tune these parameters:

```python
custom_config = {
    "n_gpu_layers": -1,     # Use all layers on GPU
    "n_ctx": 8192,          # Context window size
    "n_batch": 512,         # Batch size for prompt processing
    "n_threads": 8,         # Number of CPU threads
    "temperature": 0.7,     # Higher values = more creative output
    "top_p": 0.9,           # Controls diversity
    "top_k": 40,            # Controls vocabulary selection
    "repeat_penalty": 1.1   # Discourages repetition
}

model, config = ModelManager.load_model("path/to/model.gguf", custom_config)
```

## GPU Requirements

- CUDA 12.6 or newer is recommended for best performance
- At least 6GB VRAM for 7B models with Q4_K_M quantization
- More VRAM needed for larger models or higher quantization levels