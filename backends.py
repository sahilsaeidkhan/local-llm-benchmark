"""Backend adapters for the local LLM benchmark.

Three backends share one interface:

    available()  -> bool
    list_models() -> [str]
    generate(model, prompt, max_tokens, num_ctx, temperature, seed) -> dict

The dict returned by generate() has the same keys for every backend:
    text, prompt_tokens, output_tokens, ttft_s, total_s,
    gen_tps, prefill_tps, load_s, simulated

Where the numbers come from (checked against the official docs):

- Ollama: POST /api/generate with stream=true. The final chunk carries
  total_duration, load_duration, prompt_eval_count, prompt_eval_cached_count,
  prompt_eval_duration, eval_count and eval_duration, all durations in
  nanoseconds. Generation speed = eval_count / eval_duration * 1e9.
  Prefill speed uses (prompt_eval_count - prompt_eval_cached_count), because
  prompt_eval_duration only times the uncached part. TTFT is not in the API,
  so it is measured on the client from the stream.
  VRAM comes from GET /api/ps (size_vram per loaded model).

- llama.cpp server: POST /completion with stream=true. The final chunk
  carries tokens_predicted / tokens_evaluated and a timings object with
  prompt_per_second and predicted_per_second. TTFT is measured on the
  client from the stream, same as Ollama.

- Demo: no server at all. It sleeps in small steps to imitate a model
  running at roughly 50 tokens/sec and returns canned text. Every result
  is flagged simulated=True and the model name says SIMULATED. Never
  read demo numbers as real measurements.
"""

import hashlib
import json
import random
import time
import urllib.request
import urllib.error

OLLAMA_URL = "http://localhost:11434"
LLAMACPP_URL = "http://localhost:8080"


def _http_json(url, payload=None, timeout=5):
    """Tiny JSON-over-HTTP helper so the tool needs no extra deps."""
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _approx_tokens(text):
    """Rough token estimate, only used by the demo backend."""
    return max(1, round(len(text) / 4))


class OllamaBackend:
    name = "ollama"
    simulated = False

    def __init__(self, base_url=OLLAMA_URL):
        self.base_url = base_url.rstrip("/")

    def available(self):
        try:
            _http_json(self.base_url + "/api/tags", timeout=2)
            return True
        except Exception:
            return False

    def list_models(self):
        try:
            data = _http_json(self.base_url + "/api/tags", timeout=5)
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def vram_mb(self, model=None):
        """VRAM of loaded models from /api/ps (size_vram), in MB."""
        try:
            data = _http_json(self.base_url + "/api/ps", timeout=5)
            total = 0
            for m in data.get("models", []):
                if model is None or m.get("name") == model:
                    total += m.get("size_vram", 0)
            return round(total / 1e6, 1) if total else None
        except Exception:
            return None

    def generate(self, model, prompt, max_tokens, num_ctx, temperature=0.0, seed=42):
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_ctx": num_ctx,
                "num_predict": max_tokens,
                "temperature": temperature,
                "seed": seed,
            },
        }
        req = urllib.request.Request(
            self.base_url + "/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        ttft = None
        parts = []
        final = {}
        with urllib.request.urlopen(req, timeout=900) as resp:
            for raw in resp:
                line = raw.strip()
                if not line:
                    continue
                chunk = json.loads(line.decode("utf-8"))
                if chunk.get("response"):
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    parts.append(chunk["response"])
                if chunk.get("done"):
                    final = chunk
        total = time.perf_counter() - start

        eval_count = final.get("eval_count", 0)
        eval_dur = final.get("eval_duration", 0)
        pe_count = final.get("prompt_eval_count", 0)
        pe_cached = final.get("prompt_eval_cached_count", 0) or 0
        pe_dur = final.get("prompt_eval_duration", 0)

        gen_tps = (eval_count / eval_dur * 1e9) if eval_dur else None
        uncached = max(pe_count - pe_cached, 0)
        prefill_tps = (uncached / pe_dur * 1e9) if pe_dur and uncached else None
        load_s = (final.get("load_duration", 0) or 0) / 1e9

        return {
            "text": "".join(parts),
            "prompt_tokens": pe_count,
            "output_tokens": eval_count,
            "ttft_s": ttft if ttft is not None else total,
            "total_s": total,
            "gen_tps": gen_tps,
            "prefill_tps": prefill_tps,
            "load_s": load_s,
            "simulated": False,
        }


class LlamaCppBackend:
    name = "llamacpp"
    simulated = False

    def __init__(self, base_url=LLAMACPP_URL):
        self.base_url = base_url.rstrip("/")

    def available(self):
        try:
            _http_json(self.base_url + "/health", timeout=2)
            return True
        except Exception:
            return False

    def list_models(self):
        try:
            data = _http_json(self.base_url + "/v1/models", timeout=5)
            ids = [m["id"] for m in data.get("data", [])]
            if ids:
                return [i.replace("\\", "/").split("/")[-1] for i in ids]
        except Exception:
            pass
        try:
            props = _http_json(self.base_url + "/props", timeout=5)
            path = props.get("model_path", "")
            if path:
                return [path.replace("\\", "/").split("/")[-1]]
        except Exception:
            pass
        return ["loaded-model"]

    def _templated_prompt(self, prompt):
        """Wrap the prompt in the model's own chat template.

        /completion is a raw completion endpoint. An instruct model fed
        raw text often answers with an immediate end-of-turn token, which
        makes the benchmark measure nothing. The server exposes
        /apply-template, which renders its loaded model's template, so
        use that and fall back to the raw prompt if it is unavailable.
        """
        try:
            data = _http_json(
                self.base_url + "/apply-template",
                {"messages": [{"role": "user", "content": prompt}]},
                timeout=10,
            )
            return data.get("prompt") or prompt
        except Exception:
            return prompt

    def generate(self, model, prompt, max_tokens, num_ctx, temperature=0.0, seed=42):
        # The server serves whichever model it was started with; the model
        # name is recorded for the report but not sent.
        payload = {
            "prompt": self._templated_prompt(prompt),
            "n_predict": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "stream": True,
            "cache_prompt": False,
        }
        req = urllib.request.Request(
            self.base_url + "/completion",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        ttft = None
        parts = []
        final = {}
        with urllib.request.urlopen(req, timeout=900) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                chunk = json.loads(line[5:].strip())
                if chunk.get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    parts.append(chunk["content"])
                if chunk.get("stop"):
                    final = chunk
        total = time.perf_counter() - start

        timings = final.get("timings", {}) or {}
        output_tokens = final.get("tokens_predicted", timings.get("predicted_n", 0))
        prompt_tokens = final.get("tokens_evaluated", timings.get("prompt_n", 0))
        gen_tps = timings.get("predicted_per_second")
        prefill_tps = timings.get("prompt_per_second")
        if gen_tps is None and output_tokens and ttft is not None and total > ttft:
            gen_tps = output_tokens / (total - ttft)

        return {
            "text": "".join(parts),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "ttft_s": ttft if ttft is not None else total,
            "total_s": total,
            "gen_tps": gen_tps,
            "prefill_tps": prefill_tps,
            "load_s": None,
            "simulated": False,
        }


_DEMO_FRAGMENTS = [
    "The main idea is to keep the approach simple and measure each step. ",
    "First, look at the input carefully and note the constraints. ",
    "Then work through the logic one piece at a time, checking as you go. ",
    "A common mistake here is to rush the first step and miss an edge case. ",
    "In practice, the clean solution is usually shorter than expected. ",
    "This matters because small errors early on compound later. ",
    "To verify, test with a small example and trace the values by hand. ",
    "Finally, summarise the result and state the assumption you relied on. ",
]


class DemoBackend:
    """Simulated backend. Timing is acted out with sleeps so the shape of
    the results looks like a real run, but nothing is measured. Every
    result is flagged simulated and the model name says SIMULATED."""

    name = "demo"
    simulated = True
    MODEL = "simulated-4b-q4_k_m (SIMULATED)"

    def available(self):
        return True

    def list_models(self):
        return [self.MODEL]

    def generate(self, model, prompt, max_tokens, num_ctx, temperature=0.0, seed=42):
        digest = hashlib.sha256((prompt + str(seed)).encode("utf-8")).hexdigest()
        rng = random.Random(int(digest[:8], 16))

        prompt_tokens = _approx_tokens(prompt)
        target_tokens = max(24, int(max_tokens * rng.uniform(0.55, 0.95)))
        prefill_rate = rng.uniform(1400, 2600)   # tokens/sec, prompt processing
        decode_rate = rng.uniform(43, 58)        # tokens/sec, generation

        start = time.perf_counter()
        # Prompt processing, then the first token.
        time.sleep(min(prompt_tokens / prefill_rate, 0.4))
        step_tokens = 10
        first_step_s = step_tokens / decode_rate
        time.sleep(first_step_s)
        ttft = time.perf_counter() - start

        # Build text of roughly target_tokens tokens.
        words_target = int(target_tokens * 0.75)
        words = []
        i = rng.randrange(len(_DEMO_FRAGMENTS))
        while len(words) < words_target:
            words.extend(_DEMO_FRAGMENTS[i % len(_DEMO_FRAGMENTS)].split())
            i += 1
        text = " ".join(words[:words_target])

        remaining = max(target_tokens - step_tokens, 0)
        decode_start = time.perf_counter()
        steps = max(int(remaining / step_tokens), 0)
        for _ in range(steps):
            time.sleep(step_tokens / decode_rate)
        decode_s = time.perf_counter() - decode_start + first_step_s
        total = time.perf_counter() - start

        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "output_tokens": target_tokens,
            "ttft_s": ttft,
            "total_s": total,
            "gen_tps": target_tokens / decode_s if decode_s else None,
            "prefill_tps": prefill_rate,
            "load_s": 0.0,
            "simulated": True,
        }


BACKENDS = {
    "ollama": OllamaBackend,
    "llamacpp": LlamaCppBackend,
    "demo": DemoBackend,
}


def detect_backend(preference="auto"):
    """Return (backend_instance, note). In auto mode, try real servers
    first and fall back to the demo backend."""
    if preference != "auto":
        return BACKENDS[preference](), "selected manually"
    for key in ("ollama", "llamacpp"):
        backend = BACKENDS[key]()
        if backend.available():
            return backend, f"auto-detected at {backend.base_url}"
    return DemoBackend(), "no local server found, using the demo backend"
