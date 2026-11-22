"""Render a comparison chart from a benchmark results CSV.

Usage:
    python3 chart.py results/results-2026-11-22-1630.csv [output.png]

The chart has two panels: median generation tokens/sec per prompt
category, and median time to first token per backend/model. If any row
in the CSV is simulated, the title says so in plain words.
"""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key in ("gen_tps", "ttft_s", "total_s", "output_tokens"):
                try:
                    row[key] = float(row[key]) if row.get(key) not in ("", None) else None
                except ValueError:
                    row[key] = None
            row["simulated"] = str(row.get("simulated", "")).lower() == "true"
            rows.append(row)
    return rows


def make_chart(csv_path, png_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in load_rows(csv_path) if not r.get("error")]
    if not rows:
        raise ValueError("no successful rows in CSV, nothing to chart")

    simulated = any(r["simulated"] for r in rows)
    groups = sorted({(r["backend"], r["model"]) for r in rows})
    categories = sorted({r["category"] for r in rows})

    # Median gen tok/s per (group, category)
    by_gc = defaultdict(list)
    by_g_ttft = defaultdict(list)
    for r in rows:
        by_gc[((r["backend"], r["model"]), r["category"])].append(r["gen_tps"])
        by_g_ttft[(r["backend"], r["model"])].append(r["ttft_s"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    width = 0.8 / max(len(groups), 1)
    x = list(range(len(categories)))
    for gi, group in enumerate(groups):
        vals = [_median(by_gc[(group, c)]) or 0 for c in categories]
        offset = (gi - (len(groups) - 1) / 2) * width
        axes[0].bar([i + offset for i in x], vals, width=width,
                    label=f"{group[0]}: {group[1]}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].set_ylabel("median generation tokens/sec")
    axes[0].set_title("Generation speed by prompt category")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    labels = [f"{g[0]}\n{g[1]}" for g in groups]
    ttfts = [_median(by_g_ttft[g]) or 0 for g in groups]
    axes[1].bar(labels, ttfts, color="#4C78A8")
    axes[1].set_ylabel("median time to first token (s)")
    axes[1].set_title("Time to first token")
    axes[1].grid(axis="y", alpha=0.3)

    title = "Local LLM benchmark"
    if simulated:
        title += " — SIMULATED demo data, not real measurements"
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    if png_path is None:
        png_path = str(Path(csv_path).with_suffix(".png"))
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    return png_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 chart.py <results.csv> [output.png]")
        raise SystemExit(2)
    out = make_chart(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(f"chart written to {out}")
