#!/usr/bin/env python3
"""
orchestrator.py

Orchestrates the pipeline:
Classical NLP -> Retrieve -> Summarize -> Verify
If verifier confidence < threshold (default 0.7), the Retrieve->Summarize->Verify loop is retried with different retriever alphas.

Place this file in your project root and run:
python3 orchestrator.py --csv /path/to/news_corpus.csv --query "man riding horse"

Assumptions:
- retriever_hybrid.py, summarizer_agent.py, verifier_agent.py, classical_agent.py are runnable via python3 and produce
  results/retriever_output.json, results/generated_summary.json, results/metrics.json respectively.
"""

import argparse
import subprocess
import json
import os
import sys
import time
from typing import List

# Helper to run subprocess and stream output
def run_command(cmd: List[str], cwd: str = None, env=None, timeout: int = None):
    """Run a command and print stdout/stderr in real time. Returns (returncode, stdout, stderr)."""
    print(f"\n>>> Running: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout_lines = []
    stderr_lines = []
    # stream output
    while True:
        out = proc.stdout.readline()
        err = proc.stderr.readline()
        if out:
            print(out, end="")
            stdout_lines.append(out)
        if err:
            print(err, end="", file=sys.stderr)
            stderr_lines.append(err)
        if out == "" and err == "" and proc.poll() is not None:
            break
    rc = proc.returncode
    # read remaining
    remaining_out = proc.stdout.read()
    remaining_err = proc.stderr.read()
    if remaining_out:
        print(remaining_out, end="")
        stdout_lines.append(remaining_out)
    if remaining_err:
        print(remaining_err, end="", file=sys.stderr)
        stderr_lines.append(remaining_err)
    return rc, "".join(stdout_lines), "".join(stderr_lines)


def load_metrics(metrics_path: str):
    if not os.path.isfile(metrics_path):
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_confidence_from_metrics(metrics: dict) -> float:
    """
    Compute a composite confidence score in [0,1].
    Heuristic:
      - factual_precision (if present) contributes directly (0..1)
      - contradiction_rate subtracts (1 - contradiction_rate)
      - temporal inconsistency subtracts (1 - temporal_inconsistency_rate)
    If any metric is missing, we weigh others more heavily.

    Formula used:
      components = []
      if factual_precision available -> add factual_precision
      if contradiction_rate available -> add (1 - contradiction_rate)
      if temporal_inconsistency_rate available -> add (1 - temporal_inconsistency_rate)
      confidence = mean(components)
    """
    comps = []
    if metrics.get("factual_precision") is not None:
        comps.append(float(metrics["factual_precision"]))
    if metrics.get("contradiction_rate") is not None:
        comps.append(max(0.0, 1.0 - float(metrics["contradiction_rate"])))
    if metrics.get("temporal_inconsistency_rate") is not None:
        # lower inconsistency -> higher score
        comps.append(max(0.0, 1.0 - float(metrics["temporal_inconsistency_rate"])))
    if not comps:
        return 0.0
    return sum(comps) / len(comps)


def main():
    parser = argparse.ArgumentParser("Pipeline Orchestrator")
    parser.add_argument("--csv", type=str, default="data/news_corpus.csv", help="Path to corpus CSV")
    parser.add_argument("--query", type=str, default="man riding horse", help="Query to use for retriever")
    parser.add_argument("--k", type=int, default=5, help="Number of top results to retrieve")
    parser.add_argument("--alpha-list", nargs="+", type=float, default=[0.5, 0.3, 0.7], help="List of alpha values to try for retriever (dense weight).")
    parser.add_argument("--confidence-threshold", type=float, default=0.7, help="Retry until confidence >= threshold or alpha-list exhausted")
    parser.add_argument("--max-retries", type=int, default=2, help="Max number of retries (overall).")
    parser.add_argument("--workdir", type=str, default=".", help="Working directory (where scripts live)")
    args = parser.parse_args()

    workdir = args.workdir
    csv_path = args.csv
    query = args.query
    k = args.k
    alpha_list = args.alpha_list
    conf_thr = args.confidence_threshold
    max_retries = args.max_retries

    # Step 0: optional Classical NLP pre-processing
    classical_script = os.path.join(workdir, "src", "classical_agent.py")
    if os.path.isfile(classical_script):
        print("Step 0: Running classical NLP preprocessing...")
        rc, out, err = run_command([sys.executable, classical_script], cwd=workdir)
        if rc != 0:
            print("[Warning] classical_agent failed or exited non-zero; continuing. stderr:")
            print(err)
    else:
        print("Step 0: classical_agent.py not found — skipping classical preprocessing.")

    # Main loop: try combinations of retriever alpha values until satisfied
    attempt = 0
    used_alphas = []
    last_metrics = None
    success = False

    for retry_idx in range(max_retries + 1):
        # choose alpha for this run
        alpha = alpha_list[retry_idx % len(alpha_list)]
        used_alphas.append(alpha)
        attempt += 1
        print("\n" + "=" * 70)
        print(f"Attempt #{attempt} — retriever alpha={alpha}")
        print("=" * 70 + "\n")

        # 1) Run retriever to create results/retriever_output.json
        retriever_script = os.path.join(workdir, "src", "retriever_hybrid.py")
        if not os.path.isfile(retriever_script):
            print(f"[ERROR] retriever_hybrid.py not found at {retriever_script}. Aborting.")
            return 1

        retr_cmd = [
            sys.executable,
            retriever_script,
            "--csv", csv_path,
            "--query", query,
            "--k", str(k),
            "--alpha", str(alpha)
        ]
        rc, _, _ = run_command(retr_cmd, cwd=workdir)
        if rc != 0:
            print("[ERROR] Retriever failed. Aborting attempt.")
            continue

        # 2) Run summarizer (consumes results/retriever_output.json)
        summarizer_script = os.path.join(workdir, "src", "summarizer_agent.py")
        if not os.path.isfile(summarizer_script):
            summarizer_script = os.path.join(workdir, "src", "summarizer_qa_agent.py")
        if not os.path.isfile(summarizer_script):
            print(f"[ERROR] summarizer script not found. Aborting.")
            return 1

        sum_cmd = [
            sys.executable,
            summarizer_script,
            "--input", "retriever",
            "--retriever-path", os.path.join("results", "retriever_output.json"),
            "--model", "google/flan-t5-base",   # lighter default for faster runs (change as needed)
            "--num-claims", "5",
            "--top-k", str(max(10, k)),
            "--device", "cpu"
        ]
        rc, _, _ = run_command(sum_cmd, cwd=workdir)
        if rc != 0:
            print("[ERROR] Summarizer failed. Aborting attempt.")
            continue

        # 3) Run verifier
        verifier_script = os.path.join(workdir, "src", "verifier_agent.py")
        if not os.path.isfile(verifier_script):
            print(f"[ERROR] verifier_agent.py not found. Aborting.")
            return 1

        ver_cmd = [
            sys.executable,
            verifier_script,
            "--claims", os.path.join("results", "generated_summary.json"),
            "--retriever", os.path.join("results", "retriever_output.json"),
            "--sim-threshold", "0.8",
            "--nli-threshold", "0.5",
            "--device", "cpu",
            "--out-json", os.path.join("results", "metrics.json")
        ]
        rc, _, _ = run_command(ver_cmd, cwd=workdir)
        if rc != 0:
            print("[ERROR] Verifier failed. Aborting attempt.")
            continue

        # 4) Load metrics and compute composite confidence
        metrics = load_metrics(os.path.join("results", "metrics.json"))
        if not metrics:
            print("[ERROR] metrics.json not found or unreadable. Aborting.")
            continue

        confidence = compute_confidence_from_metrics(metrics)
        print(f"\nVerifier metrics loaded. Composite confidence: {confidence:.3f}")
        last_metrics = metrics

        if confidence >= conf_thr:
            print(f"Confidence {confidence:.3f} >= threshold {conf_thr}. Pipeline succeeded.")
            success = True
            break
        else:
            print(f"Confidence {confidence:.3f} < threshold {conf_thr}. Will retry with next alpha (if any).")
            # small wait before retry
            time.sleep(1)

        # if we've used all alpha candidates, stop early
        if retry_idx + 1 >= len(alpha_list):
            print("All configured alphas tried for this attempt loop.")
            # will continue outer retry loop if any left
            pass

    # final report
    print("\n" + "=" * 70)
    print("ORCHESTRATION COMPLETE")
    print(f"Attempts: {attempt}, Used alphas: {used_alphas}")
    if last_metrics:
        print(f"Last composite confidence: {compute_confidence_from_metrics(last_metrics):.3f}")
        print(f"Last metrics saved at: results/metrics.json")
    if success:
        print("Pipeline achieved acceptable confidence.")
        return 0
    else:
        print("Pipeline did not reach confidence threshold. Check results/metrics.json for details.")
        return 2

if __name__ == "__main__":
    sys.exit(main())
