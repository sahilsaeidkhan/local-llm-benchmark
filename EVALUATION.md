# Evaluation

Tested on 22 Nov 2026 on the build machine: a Linux VM with shared
vCPUs (4 threads used), about 8 GB RAM, and no GPU. That is a very
different machine from an RTX 3050 setup, so treat the numbers as
proof the measurement works, not as numbers to compare against a
real gaming GPU.

## What actually ran

- Backend: llama.cpp server, build b12631 (official CPU build),
  started with a 4096 token context and 4 threads.
- Model: SmolLM2-135M-Instruct, Q4_K_M GGUF (105 MB). A small model
  was chosen on purpose: this box has no GPU and little RAM, and the
  point of the test was the measurement pipeline, not the model.
  The tool itself does not care which model the server has loaded.
- Settings: temperature 0, seed 42, one warmup run, then the full
  10-prompt set from prompts.json.
- The demo backend was also run end to end (see below).
- Ollama was not installed on this machine, so the Ollama path is
  written against the documented API but was not exercised live.
  That is the one gap in this evaluation.

## Real results (llama.cpp backend)

File: results/results-20261122-112744-llamacpp.csv

| Prompt | Category | Out tokens | Gen tok/s | TTFT (s) | Total (s) | Prefill tok/s |
|---|---|---|---|---|---|---|
| code-01 | coding | 220 | 41.4 | 0.39 | 5.67 | 176 |
| code-02 | coding | 254 | 37.8 | 0.33 | 7.02 | 257 |
| code-03 | coding | 260 | 44.6 | 0.50 | 6.31 | 187 |
| reas-01 | reasoning | 195 | 30.4 | 0.61 | 6.99 | 124 |
| reas-02 | reasoning | 260 | 28.2 | 0.91 | 10.11 | 79 |
| reas-03 | reasoning | 240 | 28.2 | 0.88 | 9.30 | 97 |
| hing-01 | hinglish | 220 | 26.1 | 0.80 | 9.20 | 112 |
| hing-02 | hinglish | 260 | 29.1 | 0.90 | 9.80 | 102 |
| long-01 | long-form | 420 | 19.8 | 0.90 | 22.04 | 87 |
| long-02 | long-form | 420 | 19.7 | 0.49 | 21.71 | 171 |

Medians across the set: 28.6 generation tokens/sec, 0.71 s time to
first token, 9.25 s total latency, 118 prefill tokens/sec. Server
RAM (llama-server RSS, via psutil) grew from 256 MB to 319 MB across
the run as the KV cache filled. VRAM columns are empty because the
machine has no GPU.

Two patterns worth noting:

- Long-form prompts ran at about half the speed of coding prompts
  (19.8 vs 41 tok/s median). On CPU, decode slows as the context
  grows, and the long-form prompts both read and write more.
- Speeds are consistent within a run but move a lot between runs on
  this shared VM. An earlier run the same morning had code-03 at
  10.9 tok/s against 44.6 tok/s in the final run. Same model, same
  settings, different neighbours on the host. This is exactly why
  the tool supports --repeats and reports medians.

## Demo backend results (simulated)

File: results/results-20261122-110941-simulated.csv

Full pass completed with the demo backend: median 53.3 tokens/sec,
median TTFT 0.22 s, prefill 1870 tokens/sec. These numbers are
simulated by design (the demo imitates a model at roughly 50
tokens/sec) and exist to prove the CSV, summary, and chart flow.
They are labelled SIMULATED in the console output, the CSV's
simulated column, the model name, and the chart title.

## Problems found while testing

1. Raw prompts against /completion measure nothing. The first real
   run fed prompts to llama.cpp's /completion endpoint as plain
   text. The instruct model answered most prompts with a single
   end-of-turn token, so 7 of 10 rows showed 1 output token. The
   fix: prompts now go through the server's /apply-template endpoint
   first, so the model's own chat template is applied. After the
   fix, all 10 prompts generated full answers. That first run's
   files were deleted from results/ so nobody quotes them by
   mistake; this note is the record of it.
2. One row in an interrupted run recorded 0 output tokens although
   text came back (the run was cut short when the host service
   restarted). It did not reproduce in the clean run. If the server
   dies mid-run, the remaining prompts are recorded as error rows
   and the CSV is still written, which is the behaviour I wanted.
3. Model downloads over this network kept truncating (a 491 MB
   Qwen2.5-0.5B GGUF failed three times, including one corrupted
   resume). That is an environment problem, not a tool problem, and
   it is why the smaller SmolLM2 file was used for the test.

## Not tested

- Ollama backend live (not installed here). The timing maths follows
  the official API docs: eval_count / eval_duration for generation,
  cached tokens subtracted for prefill, size_vram from /api/ps for
  VRAM. It needs one live pass on a machine with Ollama to be fully
  proven.
- The Streamlit UI (app.py) is syntax-checked but was not run here;
  Streamlit is not installed on this box. The CLI is the tested path.
- GPU VRAM reporting (no GPU here). The code reads Ollama's /api/ps
  or nvidia-smi when present and leaves the column blank otherwise.

## Bottom line

The pipeline works end to end with a real server: detection, warmup,
templated prompts, streamed timing, CSV, chart, and medians. The
numbers above are real measurements of a small model on a weak CPU
box. On a proper GPU machine the same commands should produce the
same shape of report with much bigger numbers.
