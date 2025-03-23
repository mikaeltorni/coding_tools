
import requests
import json

payload = {
    "prompt": "Write a very long poem about a cat",
    "max_tokens": 4000
}
response = requests.post("http://localhost:8080/generate", json=payload)
print(json.dumps(response.json(), indent=2))
