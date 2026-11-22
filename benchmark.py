#!/usr/bin/env python3
"""Local LLM benchmark.

Runs one fixed prompt set (prompts.json) against a local model server
and records, for every prompt: generation tokens/sec, time to first
token, total latency, prompt and output token counts, context size,
and VRAM / server RAM where the machine exposes them.

Backends: Ollama (localhost:11434), llama.cpp server (localhost:8080),
or a built-in demo backend that simulates a model so the tool runs
anywhere. Demo numbers are labelled SIMULATED everywhere they appear.

    python3 benchmark.py                     # auto-detect backend
    python3 benchmark.py --backend llamacpp --model qwen.gguf
    python3 benchmark.py --backend demo      # force the simulation
    python3 benchmark.py --repeats 3         # median of 3 runs

Results land in results/ as a timestamped CSV plus a chart PNG.
"""

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from backends import detect_backend

try:
    import psutil
except ImportError:
    psutil = None

FIELDS = [
    "timestamp", "backend", "model", "simulated", "prompt_id", "category",
    "repeat", "prompt_tokens", "output_tokens", "ttft_s", "total_s",
    "gen_tps", "prefill_tps", "load_s", "num_ctx", "vram_mb",
    "server_ram_mb", "output_chars", "error",
]


def gpu_vram_mb():
    """Total VRAM in use on the first NVIDIA GPU, or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return float(out.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return None


def server_ram_mb(backend_name):
    """RSS of the local server process (ollama / llama-server), or None."""
    if psutil is None:
        return None
    needle = {"ollama": "ollama", "llamacpp": "llama-server"}.get(backend_name)
    if not needle:
        return None
    total = 0
    try:
        for proc in psutil.process_iter(["name", "memory_info"]):
            if needle in (proc.info["name"] or ""):
                total += proc.info["memory_info"].rss
    except Exception:
        return None
    return round(total / 1e6, 1) if total else None


def measure_vram(backend, model):
    if backend.name == "ollama":
        vram = backend.vram_mb(model)
        if vram is not None:
            return vram
    return gpu_vram_mb()


def fmt(value, digits=2):
    return "" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description="Benchmark a local LLM on a fixed prompt set.")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "ollama", "llamacpp", "demo"])
    parser.add_argument("--model", default=None,
                        help="model name (default: first one the server lists)")
    parser.add_argument("--prompts", default="prompts.json")
    parser.add_argument("--num-ctx", type=int, default=4096,
                        help="context size to request (default 4096; Mayank's setup used 16384)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="override the per-prompt output token cap")
    parser.add_argument("--repeats", type=int, default=1,
                        help="run each prompt this many times (default 1)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="fixed at 0 by default so runs are comparable")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--no-chart", action="store_true")
    args = parser.parse_args()

    backend, note = detect_backend(args.backend)
    models = backend.list_models()
    if not models:
        print(f"backend '{backend.name}' is reachable but lists no models. "
              "Pull or load a model first.")
        return 1
    model = args.model or models[0]

    print(f"backend : {backend.name} ({note})")
    print(f"model   : {model}")
    print(f"context : {args.num_ctx} tokens, temperature {args.temperature}, "
          f"seed {args.seed}, repeats {args.repeats}")
    if backend.simulated:
        print()
        print("!" * 64)
        print("  DEMO BACKEND: no local model server was found, so these")
        print("  numbers are SIMULATED. They show the shape of a report,")
        print("  not the speed of any real model.")
        print("!" * 64)

    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))

    # One warmup run so model load time does not pollute the first prompt.
    print("\nwarmup (not recorded)...")
    try:
        backend.generate(model, "Say hello in one word.", 8,
                         args.num_ctx, args.temperature, args.seed)
    except Exception as exc:
        print(f"warmup failed: {exc}")
        if not backend.simulated:
            return 1

    rows = []
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for item in prompts:
        cap = args.max_tokens or item.get("max_tokens", 256)
        for rep in range(1, args.repeats + 1):
            row = {k: "" for k in FIELDS}
            row.update({
                "timestamp": stamp, "backend": backend.name, "model": model,
                "simulated": str(backend.simulated).lower(),
                "prompt_id": item["id"], "category": item["category"],
                "repeat": rep, "num_ctx": args.num_ctx,
            })
            try:
                res = backend.generate(model, item["prompt"], cap,
                                       args.num_ctx, args.temperature, args.seed)
                row.update({
                    "prompt_tokens": res["prompt_tokens"],
                    "output_tokens": res["output_tokens"],
                    "ttft_s": fmt(res["ttft_s"], 3),
                    "total_s": fmt(res["total_s"], 3),
                    "gen_tps": fmt(res["gen_tps"]),
                    "prefill_tps": fmt(res["prefill_tps"]),
                    "load_s": fmt(res["load_s"], 3),
                    "vram_mb": fmt(measure_vram(backend, model), 1),
                    "server_ram_mb": fmt(server_ram_mb(backend.name), 1),
                    "output_chars": len(res["text"]),
                })
                print(f"  {item['id']:<8} {item['category']:<10} "
                      f"out {res['output_tokens']:>4} tok  "
                      f"{(res['gen_tps'] or 0):6.1f} tok/s  "
                      f"ttft {(res['ttft_s'] or 0):5.2f}s  "
                      f"total {res['total_s']:6.2f}s")
            except Exception as exc:
                row["error"] = str(exc)[:200]
                print(f"  {item['id']:<8} ERROR: {exc}")
            rows.append(row)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = "simulated" if backend.simulated else backend.name
    csv_path = outdir / f"results-{datetime.now():%Y%m%d-%H%M%S}-{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written to {csv_path}")

    ok_rows = [r for r in rows if not r["error"]]
    if ok_rows:
        def med(key):
            vals = [float(r[key]) for r in ok_rows if r[key] != ""]
            return statistics.median(vals) if vals else None
        print("\nsummary (median across prompts):")
        print(f"  generation speed : {med('gen_tps'):.1f} tokens/sec")
        print(f"  time to 1st token: {med('ttft_s'):.2f} s")
        print(f"  total latency    : {med('total_s'):.2f} s")
        print(f"  prefill speed    : {med('prefill_tps'):.0f} tokens/sec")
        if backend.simulated:
            print("  (all simulated — demo backend)")

    if not args.no_chart and ok_rows:
        from chart import make_chart
        png = make_chart(str(csv_path))
        print(f"chart written to {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
