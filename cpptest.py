import requests
import json

payload = {
    "prompt": "Write a very long poem about a cat",
    "n_predict": 4000
}
response = requests.post("http://localhost:8080/completion", json=payload)
print(json.dumps(response.json(), indent=2))
