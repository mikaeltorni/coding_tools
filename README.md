# Gemma 1B Model Setup Documentation

This guide explains how to set up your environment to download and run the Gemma 1B model from Hugging Face. Follow these steps:

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

---

## 3. Configure Hugging Face Authentication

Since the Gemma model is hosted in a gated repository, you need to authenticate with Hugging Face:

1. **Log in or Sign Up on Hugging Face:**  
   Visit [Hugging Face](https://huggingface.co/) and either log in or create a new account.

2. **Model Access:**  
   Ensure your Hugging Face account has been granted access to the gated Gemma 1B model. You can check the model page for any access instructions: [Gemma 1B Model](https://huggingface.co/google/gemma-3-1b-it).

3. **Generate an Access Token:**  
   - Click on your profile picture and go to **Settings**.
   - Navigate to the **Access Tokens** section.
   - Click on **New token**, give it a name, and set the required scope (usually "read").
   - Copy the generated token.

4. **Authenticate Using the CLI (Optional):**  
   Since the Hugging Face CLI is installed, log in (with the same conda environment that you installed requirements.txt to):

   ```bash
   huggingface-cli login
   ```

   Paste your token when prompted.  
   *If you prefer not to use the CLI, you can pass the token directly in your code (see the next step).*

---

## Additional Notes

- **Model Access:**  
  Ensure your Hugging Face account has been granted access to the gated Gemma 1B model. You can check the model page for any access instructions: [Gemma 1B Model](https://huggingface.co/google/gemma-3-1b-it).

- **GPU Support:**  
  The example uses `torch.bfloat16` for potentially improved performance on supported hardware. Make sure your system is configured for GPU use if needed. You need at least version 12.6 of CUDA installed from Nvidia (AMD not supported right, don't have such card, figure it out).

- **First Run:**  
  The initial model download may take some time. Once downloaded, subsequent runs will use the cached version.

Following these instructions will set up your Conda environment, install necessary packages, and allow you to download and run the Gemma 1B model.