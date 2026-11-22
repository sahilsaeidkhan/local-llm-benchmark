"""Optional Streamlit UI for the local LLM benchmark.

    streamlit run app.py

Pick a backend and model, run the same fixed prompt set as the CLI,
and see the table and chart in the browser. The command line tool
(benchmark.py) stays the main entry point; this is a viewer/runner
for people who prefer clicking.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from backends import BACKENDS, detect_backend
from chart import make_chart
import benchmark as bench

st.set_page_config(page_title="Local LLM Benchmark", layout="wide")
st.title("Local LLM Benchmark")
st.caption("One fixed prompt set, measured the same way every run.")

backend_choice = st.sidebar.selectbox("Backend", ["auto", "ollama", "llamacpp", "demo"])
backend, note = detect_backend(backend_choice)
models = backend.list_models()

st.sidebar.write(f"Detected: **{backend.name}** ({note})")
if backend.simulated:
    st.warning("Demo backend: no local model server found. Every number "
               "below is SIMULATED and only shows the shape of a report.")

model = st.sidebar.selectbox("Model", models) if models else None
num_ctx = st.sidebar.number_input("Context size", 512, 32768, 4096, step=512)
repeats = st.sidebar.number_input("Repeats per prompt", 1, 5, 1)

prompts = json.loads(Path("prompts.json").read_text(encoding="utf-8"))
st.write(f"{len(prompts)} prompts loaded from prompts.json "
         f"(coding, reasoning, Hinglish, long-form).")

if st.button("Run benchmark", disabled=model is None):
    rows = []
    progress = st.progress(0.0)
    total_steps = len(prompts) * repeats
    step = 0
    for item in prompts:
        for rep in range(1, repeats + 1):
            res = backend.generate(model, item["prompt"],
                                   item.get("max_tokens", 256),
                                   int(num_ctx), 0.0, 42)
            rows.append({
                "prompt": item["id"], "category": item["category"],
                "prompt_tokens": res["prompt_tokens"],
                "output_tokens": res["output_tokens"],
                "ttft_s": round(res["ttft_s"], 3),
                "total_s": round(res["total_s"], 3),
                "gen_tps": round(res["gen_tps"], 1) if res["gen_tps"] else None,
                "prefill_tps": round(res["prefill_tps"]) if res["prefill_tps"] else None,
                "simulated": res["simulated"],
            })
            step += 1
            progress.progress(step / total_steps)
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    st.metric("Median generation speed", f"{df['gen_tps'].median():.1f} tok/s")
    st.metric("Median time to first token", f"{df['ttft_s'].median():.2f} s")

    out = Path("results/ui-run.csv")
    out.parent.mkdir(exist_ok=True)
    export = df.rename(columns={"prompt": "prompt_id"})
    export["backend"] = backend.name
    export["model"] = model
    export["error"] = ""
    export.to_csv(out, index=False)
    st.image(make_chart(str(out)), caption="Saved under results/")
