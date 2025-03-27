Setup the Llama server and run it (in main readme)

```bash
npx promptfoo@latest eval -c diff_analyzer_eval.yaml --max-concurrency 1 --repeat 20
```

# To view the results: open a new terminal and run:
```bash
npx promptfoo@latest view -y
```
(advised to open a new terminal for this, since it's a server that needs to be running all the time)