# Model Inference with llama.cpp

This guide explains how to set up your environment to run optimized inference with llama.cpp. Follow these steps:

---

## 1. Create and Activate the Conda Environment

Make sure you have [Conda](https://www.anaconda.com/docs/getting-started/miniconda/main) installed (via Anaconda or Miniconda). Create a new environment named **ct** with Python version 3.12.8:

```bash
conda create --name ct python=3.12.8
conda activate ct
```

---

## 2. GPU Acceleration Setup for Llama.cpp

### Prerequisites
- Install CUDA Toolkit 12.8 (or newer) from the official NVIDIA website.
- Verify that your NVIDIA GPU (for example, an RTX 4090 with compute capability 8.9) is supported.

### Build Instructions
1. Open a Windows Command Prompt.
2. Configure and build llama.cpp by running the following command:
```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES="89" && cmake --build build --config Release
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

## 4. Running the Llama Server

1. Open a new Command Prompt.
2. Set the CUDA_VISIBLE_DEVICES environment variable and start the server with your model by running:
```bash
set CUDA_VISIBLE_DEVICES=-0 && llama-server --model your_model.gguf
```

3. The server will start and display information about the model and inference settings.
4. Connect to the server using HTTP requests to localhost on the default port.

---

## 5. Tuning GPU Offloading

Llama.cpp automatically determines how many layers to offload to the GPU, but you can override this setting for optimal performance:

- To manually specify the number of GPU offloaded layers, add the `--n-gpu-layers` flag
- For example:
```bash
set CUDA_VISIBLE_DEVICES=-0 && llama-server --model your_model.gguf --n-gpu-layers 26
```
- If the performance is slow, increase the value. In one test, adjusting `--n-gpu-layers` to 420 significantly improved token throughput.
- Other useful parameters:
  - `--ctx-size`: Context window size (default: 2048)
  - `--batch-size`: Batch size for prompt processing
  - `--threads`: Number of CPU threads to use
  - `--stream`: Enable streaming mode

---

## 6. Monitoring and Verification

- Open a separate Command Prompt and run `nvidia-smi -l 1` to monitor GPU usage during inference
- Check the server logs to ensure that layers are being offloaded to the GPU as intended
- Look for performance metrics in the server output to verify acceleration is working properly

---

## GPU Requirements

- CUDA 12.6 or newer is recommended for best performance
- At least 6GB VRAM for 7B models with Q4_K_M quantization
- More VRAM needed for larger models or higher quantization levels