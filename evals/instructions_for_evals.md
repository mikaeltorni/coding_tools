set model=google/gemma-3-1b-it
set volume=%cd%\data
set token=hf_tokenhere

docker run --gpus all --shm-size 1g -p 8080:80 -v %volume%:/data -e HF_TOKEN=%token% ghcr.io/huggingface/text-generation-inference:3.2.1 --model-id %model%  

npx promptfoo@latest eval -c testeval.yaml --max-concurrency 1 --repeat 10 -y

# To view the results: open a new terminal and run:
```bash
npx promptfoo@latest view -y
```
(advised to open a new terminal for this, since it's a server that needs to be running all the time)