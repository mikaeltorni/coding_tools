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
- Verify that your NVIDIA GPU (for example, an RTX 4090 with compute capability 8.9 [Get the value from here, remove dot and insert it to the -DCMAKE_CUDA_ARCHITECTURES parameter](https://developer.nvidia.com/cuda-gpus)) is supported.

### Build Instructions
1. Open a Windows Command Prompt.
2. Configure and build llama.cpp by running the following commands:
```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES="89" && cmake --build build --config Release
```

---

## 3. Download a GGUF Model

1. **Download the Gemma 3 1B Model:**
   - Visit [unsloth/gemma-3-1b-it-GGUF](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/tree/main)
   - Download the Q4_K_M quantized model file (ending with .gguf extension)
   - This quantization level provides a good balance between model size and inference quality

---

## 4. Running the Llama Server

1. Open a new Command Prompt.
2. Set the CUDA_VISIBLE_DEVICES environment variable and start the server with your model by running:
```bash
set CUDA_VISIBLE_DEVICES=-0 && ..\llama.cpp\build\bin\Release\llama-server --model ..\models\gemma-3-1b-it-Q4_K_M.gguf --n-gpu-layers 420
```
(set up a high gpu layers number to increase the models inference speed)
3. The server will start and display information about the model and inference settings.
4. Connect to the server using HTTP requests to localhost on the default port.

---

EDIT THIS LATER?

- Other useful parameters:
  - `--ctx-size`: Context window size (default: 2048)
  - `--batch-size`: Batch size for prompt processing
  - `--threads`: Number of CPU threads to use
  - `--stream`: Enable streaming mode

---

## 5. Monitoring and Verification

- Open a separate Command Prompt and run `nvidia-smi -l 1` to monitor GPU usage during inference
- Check the server logs to ensure that layers are being offloaded to the GPU as intended
- Look for performance metrics in the server output to verify acceleration is working properly

---
