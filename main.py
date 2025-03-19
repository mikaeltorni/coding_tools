import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="google/gemma-3-1b-it", # "google/gemma-3-12b-it", "google/gemma-3-27b-it" 
    device="cuda",
    torch_dtype=torch.bfloat16
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Hello world."}
        ]
    }
]

output = pipe(text=messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])
