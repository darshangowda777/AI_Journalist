#!/usr/bin/env python3
"""
visualize_agent.py

Task 6 — Visualization & Presentation

Generates visualizations for:
1. BM25 vs Chroma score weights (scatter + bar)
2. LangGraph pipeline trace (Classical → Retrieve → Summarize → Verify)

Expected inputs:
- results/retriever_output.json
- results/metrics.json (optional, for confidence display)

Outputs:
- results/visualization_bm25_chroma_scatter.png
- results/visualization_bm25_chroma_bar.png
- results/visualization_pipeline_trace.png
- results/retriever_scores_snapshot.csv
- results/visualization_summary.csv
"""

import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# --- Setup ---
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

retriever_path = RESULTS_DIR / "retriever_output.json"
metrics_path = RESULTS_DIR / "metrics.json"

# --- Load or create demo retriever data ---
if retriever_path.exists():
    with open(retriever_path, "r", encoding="utf-8") as f:
        retriever_data = json.load(f)
else:
    print("[WARN] retriever_output.json not found — creating demo data.")
    retriever_data = [
        {"id": f"doc_{i}", "text": f"Document {i}", "sparse_score": (i % 5) / 4, "dense_score": ((5 - i) % 5) / 4, "hybrid_score": 0.5}
        for i in range(10)
    ]
    with open(retriever_path, "w", encoding="utf-8") as f:
        json.dump(retriever_data, f, indent=2)

# --- Convert to DataFrame ---
df = pd.DataFrame([
    {
        "doc_id": item.get("id", item.get("ID", f"doc_{i}")),
        "sparse_score": float(item.get("sparse_score", 0.0)),
        "dense_score": float(item.get("dense_score", 0.0)),
        "hybrid_score": float(item.get("hybrid_score", 0.0)),
    }
    for i, item in enumerate(retriever_data)
])

df.to_csv(RESULTS_DIR / "retriever_scores_snapshot.csv", index=False)

# --- Plot 1: Scatter of BM25 vs Chroma scores ---
plt.figure(figsize=(7, 7))
plt.scatter(df["sparse_score"], df["dense_score"], c=df["hybrid_score"], cmap="viridis", s=100, alpha=0.8)
plt.xlabel("BM25 (Sparse) Score", fontsize=12)
plt.ylabel("Chroma (Dense) Score", fontsize=12)
plt.title("BM25 vs Chroma Scores (per document)", fontsize=14)
plt.colorbar(label="Hybrid Score")
for _, row in df.iterrows():
    plt.annotate(row["doc_id"], (row["sparse_score"], row["dense_score"]), textcoords="offset points", xytext=(3,3), fontsize=8)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "visualization_bm25_chroma_scatter.png", dpi=150)
plt.close()

# --- Plot 2: Bar chart of average sparse vs dense contributions ---
avg_sparse = df["sparse_score"].mean()
avg_dense = df["dense_score"].mean()

plt.figure(figsize=(6, 4))
plt.bar(["BM25 (Sparse)", "Chroma (Dense)"], [avg_sparse, avg_dense], color=["#0072B2", "#E69F00"])
plt.ylabel("Average Normalized Score", fontsize=11)
plt.title("Average Score Contribution: BM25 vs Chroma", fontsize=13)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "visualization_bm25_chroma_bar.png", dpi=150)
plt.close()

# --- Load metrics for confidence ---
def compute_confidence(metrics: dict) -> float:
    comps = []
    if metrics.get("factual_precision") is not None:
        comps.append(float(metrics["factual_precision"]))
    if metrics.get("contradiction_rate") is not None:
        comps.append(max(0.0, 1.0 - float(metrics["contradiction_rate"])))
    if metrics.get("temporal_inconsistency_rate") is not None:
        comps.append(max(0.0, 1.0 - float(metrics["temporal_inconsistency_rate"])))
    return sum(comps) / len(comps) if comps else 0.0

metrics = {}
if metrics_path.exists():
    with open(metrics_path, "r", encoding="utf-8") as f:
        try:
            metrics = json.load(f)
        except Exception:
            pass

confidence = compute_confidence(metrics)
confidence_str = f"{confidence:.2f}"

# --- Plot 3: LangGraph Pipeline Trace ---
plt.figure(figsize=(9, 3))
ax = plt.gca()
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Node positions
nodes = {
    "Classical NLP": (10, 50),
    "Retrieve": (35, 50),
    "Summarize": (60, 50),
    "Verify": (85, 50),
}

# Draw nodes
for name, (x, y) in nodes.items():
    rect_w, rect_h = 20, 15
    ax.add_patch(plt.Rectangle((x - rect_w / 2, y - rect_h / 2), rect_w, rect_h, fill=True, color="#56B4E9", alpha=0.3))
    ax.text(x, y, name, ha="center", va="center", fontsize=10, weight="bold")

# Draw arrows
def arrow(a, b):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.5))

arrow(nodes["Classical NLP"], nodes["Retrieve"])
arrow(nodes["Retrieve"], nodes["Summarize"])
arrow(nodes["Summarize"], nodes["Verify"])

# Annotate confidence
ax.text(nodes["Verify"][0], nodes["Verify"][1] - 18, f"Composite Confidence: {confidence_str}", ha="center", fontsize=10)

plt.title("LangGraph Pipeline Trace", fontsize=12, weight="bold")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "visualization_pipeline_trace.png", dpi=150)
plt.close()

# --- Save summary CSV ---
summary_df = pd.DataFrame([
    {"metric": "avg_sparse", "value": avg_sparse},
    {"metric": "avg_dense", "value": avg_dense},
    {"metric": "composite_confidence", "value": confidence},
])
summary_df.to_csv(RESULTS_DIR / "visualization_summary.csv", index=False)

print("\n✅ Visualization complete! Files saved in 'results/' directory:")
print(" - visualization_bm25_chroma_scatter.png")
print(" - visualization_bm25_chroma_bar.png")
print(" - visualization_pipeline_trace.png")
print(" - retriever_scores_snapshot.csv")
print(" - visualization_summary.csv")
#!/usr/bin/env python3
"""
utils.py

Task 6 — Visualization & Presentation

Visualizes:
1. BM25 vs Chroma score weights (scatter + bar)
2. LangGraph pipeline trace (Classical → Retrieve → Summarize → Verify)

Inputs:
- results/retriever_output.json
- results/metrics.json (optional, for confidence display)

Outputs:
- results/visualization_bm25_chroma_scatter.png
- results/visualization_bm25_chroma_bar.png
- results/visualization_pipeline_trace.png
- results/retriever_scores_snapshot.csv
- results/visualization_summary.csv
"""

import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def visualize():
    """Run the full visualization process."""
    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)

    retriever_path = RESULTS_DIR / "retriever_output.json"
    metrics_path = RESULTS_DIR / "metrics.json"

    # --- Load or create demo retriever data ---
    if retriever_path.exists():
        with open(retriever_path, "r", encoding="utf-8") as f:
            retriever_data = json.load(f)
    else:
        print("[WARN] retriever_output.json not found — creating demo data.")
        retriever_data = [
            {
                "id": f"doc_{i}",
                "text": f"Document {i}",
                "sparse_score": (i % 5) / 4,
                "dense_score": ((5 - i) % 5) / 4,
                "hybrid_score": 0.5,
            }
            for i in range(10)
        ]
        with open(retriever_path, "w", encoding="utf-8") as f:
            json.dump(retriever_data, f, indent=2)

    # --- Convert to DataFrame ---
    df = pd.DataFrame(
        [
            {
                "doc_id": item.get("id", item.get("ID", f"doc_{i}")),
                "sparse_score": float(item.get("sparse_score", 0.0)),
                "dense_score": float(item.get("dense_score", 0.0)),
                "hybrid_score": float(item.get("hybrid_score", 0.0)),
            }
            for i, item in enumerate(retriever_data)
        ]
    )

    df.to_csv(RESULTS_DIR / "retriever_scores_snapshot.csv", index=False)

    # --- Plot 1: Scatter of BM25 vs Chroma scores ---
    plt.figure(figsize=(7, 7))
    plt.scatter(
        df["sparse_score"],
        df["dense_score"],
        c=df["hybrid_score"],
        cmap="viridis",
        s=100,
        alpha=0.8,
    )
    plt.xlabel("BM25 (Sparse) Score", fontsize=12)
    plt.ylabel("Chroma (Dense) Score", fontsize=12)
    plt.title("BM25 vs Chroma Scores (per document)", fontsize=14)
    plt.colorbar(label="Hybrid Score")
    for _, row in df.iterrows():
        plt.annotate(
            row["doc_id"],
            (row["sparse_score"], row["dense_score"]),
            textcoords="offset points",
            xytext=(3, 3),
            fontsize=8,
        )
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "visualization_bm25_chroma_scatter.png", dpi=150)
    plt.close()

    # --- Plot 2: Bar chart of average sparse vs dense contributions ---
    avg_sparse = df["sparse_score"].mean()
    avg_dense = df["dense_score"].mean()

    plt.figure(figsize=(6, 4))
    plt.bar(
        ["BM25 (Sparse)", "Chroma (Dense)"],
        [avg_sparse, avg_dense],
        color=["#0072B2", "#E69F00"],
    )
    plt.ylabel("Average Normalized Score", fontsize=11)
    plt.title("Average Score Contribution: BM25 vs Chroma", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "visualization_bm25_chroma_bar.png", dpi=150)
    plt.close()

    # --- Load metrics for confidence ---
    def compute_confidence(metrics: dict) -> float:
        comps = []
        if metrics.get("factual_precision") is not None:
            comps.append(float(metrics["factual_precision"]))
        if metrics.get("contradiction_rate") is not None:
            comps.append(max(0.0, 1.0 - float(metrics["contradiction_rate"])))
        if metrics.get("temporal_inconsistency_rate") is not None:
            comps.append(max(0.0, 1.0 - float(metrics["temporal_inconsistency_rate"])))
        return sum(comps) / len(comps) if comps else 0.0

    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            try:
                metrics = json.load(f)
            except Exception:
                pass

    confidence = compute_confidence(metrics)
    confidence_str = f"{confidence:.2f}"

    # --- Plot 3: LangGraph Pipeline Trace ---
    plt.figure(figsize=(9, 3))
    ax = plt.gca()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    nodes = {
        "Classical NLP": (10, 50),
        "Retrieve": (35, 50),
        "Summarize": (60, 50),
        "Verify": (85, 50),
    }

    for name, (x, y) in nodes.items():
        rect_w, rect_h = 20, 15
        ax.add_patch(
            plt.Rectangle(
                (x - rect_w / 2, y - rect_h / 2),
                rect_w,
                rect_h,
                fill=True,
                color="#56B4E9",
                alpha=0.3,
            )
        )
        ax.text(x, y, name, ha="center", va="center", fontsize=10, weight="bold")

    def arrow(a, b):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.5))

    arrow(nodes["Classical NLP"], nodes["Retrieve"])
    arrow(nodes["Retrieve"], nodes["Summarize"])
    arrow(nodes["Summarize"], nodes["Verify"])

    ax.text(
        nodes["Verify"][0],
        nodes["Verify"][1] - 18,
        f"Composite Confidence: {confidence_str}",
        ha="center",
        fontsize=10,
    )

    plt.title("LangGraph Pipeline Trace", fontsize=12, weight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "visualization_pipeline_trace.png", dpi=150)
    plt.close()

    # --- Save summary CSV ---
    summary_df = pd.DataFrame(
        [
            {"metric": "avg_sparse", "value": avg_sparse},
            {"metric": "avg_dense", "value": avg_dense},
            {"metric": "composite_confidence", "value": confidence},
        ]
    )
    summary_df.to_csv(RESULTS_DIR / "visualization_summary.csv", index=False)

    print("\n✅ Visualization complete! Files saved in 'results/' directory:")
    print(" - visualization_bm25_chroma_scatter.png")
    print(" - visualization_bm25_chroma_bar.png")
    print(" - visualization_pipeline_trace.png")
    print(" - retriever_scores_snapshot.csv")
    print(" - visualization_summary.csv")


# --- Run directly ---
if __name__ == "__main__":
    visualize()
