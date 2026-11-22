# Local LLM Benchmark

Run one fixed set of prompts against a local model and get back
numbers instead of vibes: generation tokens per second, time to
first token, total latency, prompt and output token counts,
context size, and VRAM / server RAM where the machine exposes
them. Results go to a CSV, with a chart PNG rendered next to it.

It exists because "this model feels fast" is not a measurement.
Run the same prompts before and after a change (a new quant, a
bigger context, a different thread count) and the CSVs tell you
what changed.

## Quick start

```
pip install -r requirements.txt
python3 benchmark.py
```

The tool looks for a running Ollama server (localhost:11434)
first, then a llama.cpp server (localhost:8080). If neither is
up, it falls back to a built-in demo backend so you can still
see the full flow. Demo numbers are simulated and are labelled
SIMULATED in the console, the CSV, the chart, and the model
name. They are never mixed with real measurements.

## One real run

Verified end to end on 22 Nov 2026 against a real llama.cpp
server (SmolLM2-135M-Instruct, Q4_K_M) on a CPU-only VM with no
GPU: all 10 prompts, no errors. Medians across the set were
28.6 generation tokens/sec, 0.71 s time to first token, 9.25 s
total latency. Small model, weak machine, real numbers. The
full table, the demo-backend run, and the problems found while
testing are in `EVALUATION.md`, and the run's CSV is in
`results/`.

## What gets measured

The prompt set lives in `prompts.json`: 10 prompts across
coding, reasoning, Hinglish, and long-form writing, each with
its own output token cap. Edit the file to use your own
prompts; the format is just id, category, prompt, max_tokens.

For the real backends the token counts and speeds come from
the servers themselves, not from client-side guessing:

- Ollama returns eval_count and eval_duration (nanoseconds) in
  the final streamed chunk. Generation speed is eval_count /
  eval_duration. Prefill speed subtracts
  prompt_eval_cached_count first, because Ollama only times the
  uncached part of the prompt. VRAM comes from /api/ps
  (size_vram).
- llama.cpp returns a timings object with prompt_per_second
  and predicted_per_second, plus tokens_predicted and
  tokens_evaluated. One catch: /completion is a raw text
  endpoint, and an instruct model fed raw text often stops
  after a token or two. So the prompt is first wrapped in the
  model's own chat template through the server's
  /apply-template endpoint. Ollama applies its template
  automatically.

Time to first token is not exposed by either API, so it is
measured on the client from the stream: time from sending the
request to the first chunk with text. One warmup run happens
before measuring, so model load time does not land on the
first prompt.

Defaults are chosen for comparability: temperature 0, seed 42,
context 4096. Override with flags:

```
python3 benchmark.py --backend llamacpp --num-ctx 16384 --repeats 3
python3 benchmark.py --backend ollama --model qwen2.5:7b
python3 benchmark.py --backend demo
```

With --repeats above 1, every run is its own CSV row and the
console summary uses medians.

## Outputs

- `results/results-<timestamp>-<backend>.csv` — one row per prompt run
- `results/results-<timestamp>-<backend>.png` — median generation
  speed by category, plus median time to first token

You can chart any older CSV again with `python3 chart.py <file.csv>`.

## Optional UI

```
streamlit run app.py
```

Same prompt set and backends, in a browser page. The CLI is
the main tool; the UI is a convenience, and it was only
syntax-checked on the build machine, not run.

## Honest limitations

- One machine, one model at a time. There is no concurrency
  testing; this measures single-stream speed, which is what a
  local coding setup actually feels like, but it says nothing
  about serving many users at once.
- The Ollama backend follows the documented API but was not
  exercised live on the build machine (Ollama was not
  installed there). The llama.cpp and demo paths were.
- Time to first token includes local network and client
  overhead. On localhost that is small, but it is not zero.
- VRAM is only as good as the source: exact from Ollama's
  /api/ps, machine-wide from nvidia-smi for llama.cpp (so it
  includes anything else on the GPU), and absent on CPU-only
  machines.
- Server RAM is the RSS of the server process found by name.
  If the server runs in Docker or under a different name, the
  column is blank.
- Token counts come from each server's own tokenizer. The same
  prompt is a different number of tokens on different models,
  so compare tokens/sec within a model family carefully.
- Speeds on a shared machine move around between runs. Use
  --repeats and read the medians.
- The demo backend imitates a model at roughly 50 tokens/sec.
  It is there to test the pipeline, not to say anything about
  real hardware.

## Files

- `benchmark.py` — CLI, detection, run loop, CSV and summary
- `backends.py` — Ollama, llama.cpp, and demo adapters
- `chart.py` — CSV to chart, usable on its own
- `app.py` — optional Streamlit UI
- `prompts.json` — the fixed prompt set
- `results/` — CSVs from the verified runs (charts are PNG, regenerate with chart.py)
- `EVALUATION.md` — results from the test run on the build machine
