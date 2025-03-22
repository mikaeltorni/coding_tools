Setup text-generation-interface (clarify url at least here later...)

Remember to get your token [FROM]
and insert it to the following command:

Configure it and run it:
```bash
set model=google/gemma-3-1b-it
set volume=%cd%\data
set token=hf_tokenhere

docker run --gpus all --shm-size 1g -p 8080:80 -v %volume%:/data -e HF_TOKEN=%token% ghcr.io/huggingface/text-generation-inference:3.2.1 --model-id %model% --quantize bitsandbytes-nf4
```

```bash
npx promptfoo@latest eval -c diff_analyzer_eval.yaml --max-concurrency 1 --repeat 10
```

# To view the results: open a new terminal and run:
```bash
npx promptfoo@latest view -y
```
(advised to open a new terminal for this, since it's a server that needs to be running all the time)