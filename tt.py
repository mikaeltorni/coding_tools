import llama_cpp
print(f"llama-cpp-python version: {llama_cpp.__version__}")
print("Available CUDA backends:")
for attr in dir(llama_cpp):
    if attr.startswith("LLAMA_BACKEND_"):
        print(f"  - {attr}")