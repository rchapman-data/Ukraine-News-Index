"""
Turn benchmark_search.py's results into the headline show-and-tell chart:
query latency and recall@10 for brute-force vs. FAISS IVF vs. FAISS HNSW,
across corpus scale (5k -> 20k -> 50k -> full ~120k articles).

Two panels, side by side:
  - Left:  average query latency (ms, log scale) vs. corpus scale
  - Right: recall@10 vs. corpus scale

The story the chart should tell at a glance: brute-force's latency should
climb roughly linearly with scale, while IVF/HNSW should stay much flatter
-- and their recall lines should show whether that speed is coming at a
real accuracy cost or not.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_PATH = Path("results/benchmark_results.csv")
CHART_PATH = Path("results/benchmark_chart.png")

METHOD_LABELS = {
    "brute_force": "Brute-force (exact)",
    "faiss_ivf": "FAISS IVF",
    "faiss_hnsw": "FAISS HNSW",
}
METHOD_ORDER = ["brute_force", "faiss_ivf", "faiss_hnsw"]


def main():
    print(f"Loading {RESULTS_PATH} ...")
    df = pd.read_csv(RESULTS_PATH)

    fig, (ax_latency, ax_recall) = plt.subplots(1, 2, figsize=(11, 4.5))

    for method in METHOD_ORDER:
        sub = df[df["method"] == method].sort_values("scale")
        ax_latency.plot(
            sub["scale"], sub["avg_query_ms"], marker="o", label=METHOD_LABELS[method]
        )
        ax_recall.plot(
            sub["scale"], sub["recall_at_10"], marker="o", label=METHOD_LABELS[method]
        )

    ax_latency.set_yscale("log")
    ax_latency.set_xlabel("Corpus size (articles)")
    ax_latency.set_ylabel("Avg. query latency (ms, log scale)")
    ax_latency.set_title("Query latency vs. corpus scale")
    ax_latency.legend()
    ax_latency.grid(True, which="both", alpha=0.3)

    ax_recall.set_xlabel("Corpus size (articles)")
    ax_recall.set_ylabel("Recall@10 (vs. brute-force)")
    ax_recall.set_title("Recall vs. corpus scale")
    ax_recall.set_ylim(0, 1.05)
    ax_recall.legend()
    ax_recall.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_PATH, dpi=150)
    print(f"Saved chart to {CHART_PATH}")


if __name__ == "__main__":
    main()
