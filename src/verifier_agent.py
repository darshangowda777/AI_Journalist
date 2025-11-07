#!/usr/bin/env python3
"""
verifier_agent.py

Verifies generated claims against source documents and computes:
- factual_precision (semantic similarity >= threshold)
- contradiction_rate (via NLI)
- temporal_consistency (date checks)

Inputs:
- results/generated_summary.json  (required)
- results/retriever_output.json or data/news_corpus.csv

Outputs:
- results/metrics.json
- results/metrics.txt

Usage:
python3 verifier_agent.py --claims results/generated_summary.json --retriever results/retriever_output.json
"""

import os
import json
import argparse
import re
from typing import List, Tuple, Dict, Any
from collections import defaultdict
from datetime import datetime

# NLP libs
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------
# Safe sentence tokenizer (same fallback as summarizer)
# ---------------------------
def sent_tokenize_safe(text: str) -> List[str]:
    if not text:
        return []
    try:
        import nltk
        from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
        return [s.strip() for s in nltk_sent_tokenize(text) if s.strip()]
    except Exception:
        import re
        pattern = r'(?<=[.!?])\s+(?=[A-Z0-9"\'“])|\n+'
        parts = [p.strip() for p in re.split(pattern, text) if p and p.strip()]
        splitted = []
        for p in parts:
            if len(p) > 1000:
                sub = [s.strip() for s in re.split(r'(?<=[.!?])\s+', p) if s.strip()]
                splitted.extend(sub)
            else:
                splitted.append(p)
        return splitted

# ---------------------------
# Load claims (structured JSON from summarizer)
# ---------------------------
def load_claims(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Claims file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Expect list of {claim_id, claim, citations: [{doc_id,snippet,score},...]}
    return data

# ---------------------------
# Load retriever docs (fallback) - create mapping doc_id -> text
# ---------------------------
def load_retriever_docs(path: str) -> Dict[str, str]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {}
    for item in data:
        doc_id = item.get("ID") or item.get("id") or ""
        # build text: prefer text field then title
        text = item.get("text") or item.get("title") or ""
        mapping[str(doc_id)] = text
    return mapping

# ---------------------------
# Date extraction helpers
# ---------------------------
# quick regex patterns for common date formats
_DATE_PATTERNS = [
    r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',            # 01-02-2020 or 1/2/20
    r'\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',              # 2020-01-02
    r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b',  # Jan 2, 2020
    r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?,?\s+\d{4}\b',   # 2 January 2020
    r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
]

def extract_dates(text: str) -> List[str]:
    found = set()
    for pat in _DATE_PATTERNS:
        for m in re.findall(pat, text, flags=re.IGNORECASE):
            found.add(m.strip())
    return list(found)

# try to parse date strings to year/month/day using dateutil if available, otherwise basic heuristics
def parse_date(date_str: str):
    try:
        from dateutil import parser as dparser
        return dparser.parse(date_str, fuzzy=True)
    except Exception:
        # fallbacks for YYYY-MM-DD or digits-only
        try:
            if re.match(r'^\d{4}-\d{1,2}-\d{1,2}$', date_str):
                return datetime.fromisoformat(date_str)
            # extract 4-digit year
            y = re.search(r'(\d{4})', date_str)
            if y:
                return datetime(int(y.group(1)), 1, 1)
        except Exception:
            return None
    return None

# ---------------------------
# Verifier core
# ---------------------------
class Verifier:
    def __init__(self, sim_model_name="sentence-transformers/all-MiniLM-L6-v2", nli_model_name="facebook/bart-large-mnli", device="cpu"):
        print("Loading sentence-transformers model for similarity:", sim_model_name)
        self.sim_model = SentenceTransformer(sim_model_name)
        print("Loading NLI model:", nli_model_name)
        # Use transformers pipeline for NLI
        try:
            # load model & tokenizer to avoid repeated loads
            tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
            model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
            self.nli = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if device=="cuda" else -1, return_all_scores=True)
        except Exception as e:
            # fallback to pipeline direct (may still work)
            self.nli = pipeline("text-classification", model=nli_model_name, device=0 if device=="cuda" else -1, return_all_scores=True)

        # thresholds (configurable via method args)
        self.sim_threshold = 0.8
        self.nli_contradiction_label = "CONTRADICTION"  # model labels may be 'contradiction', 'entailment', 'neutral' - we'll match case-insensitively

    def compute_factual_precision(self, claims: List[Dict], retriever_docs: Dict[str,str], sim_threshold: float = 0.8) -> Dict:
        """
        For each claim, evaluate its cited snippets: compute cosine similarity between claim and snippet.
        factual_precision = (# of cited snippets with sim >= threshold) / (total cited snippets)
        We'll also compute a per-claim 'supported' flag (at least one snippet >= threshold).
        """
        self.sim_threshold = sim_threshold
        total_snips = 0
        supported_snips = 0
        per_claim_support = []
        # Precompute embeddings for all snippets (gather them)
        snippets = []
        snippet_map = []  # (claim_idx, doc_id, snippet_text)
        for ci, item in enumerate(claims):
            cit_list = item.get("citations", [])
            for s in cit_list:
                snippets.append(s.get("snippet", "") if isinstance(s, dict) else s[1])
                snippet_map.append((ci, s.get("doc_id") if isinstance(s, dict) else s[0], s.get("snippet") if isinstance(s, dict) else s[1]))
        # If no snippets, return zeros
        if len(snippets) == 0:
            return {"factual_precision": None, "total_snippets": 0, "supported_snippets": 0, "per_claim": []}
        emb_snips = self.sim_model.encode(snippets, convert_to_tensor=True, show_progress_bar=False)
        # For each claim, compute embedding and compare to its snippets
        claim_embs = []
        for item in claims:
            claim_embs.append(self.sim_model.encode(item.get("claim", ""), convert_to_tensor=True, show_progress_bar=False))
        # iterate snippets
        claim_to_results = defaultdict(list)
        si = 0
        for idx, (ci, doc_id, snip_text) in enumerate(snippet_map):
            total_snips += 1
            sim = util.cos_sim(claim_embs[ci], emb_snips[idx]).item()
            supported = sim >= sim_threshold
            if supported:
                supported_snips += 1
            claim_to_results[ci].append({"doc_id": doc_id, "snippet": snip_text, "similarity": sim, "supported": supported})
            si += 1
        # per-claim supported flag
        per_claim_support = []
        for ci, item in enumerate(claims):
            res_list = claim_to_results.get(ci, [])
            supported_any = any(r["supported"] for r in res_list)
            per_claim_support.append({"claim_id": item.get("claim_id", ci+1), "claim": item.get("claim", ""), "supported_any": supported_any, "snippet_count": len(res_list), "details": res_list})
        factual_precision = supported_snips / total_snips if total_snips > 0 else None
        return {"factual_precision": factual_precision, "total_snippets": total_snips, "supported_snippets": supported_snips, "per_claim": per_claim_support}

    def compute_contradiction_rate(self, claims: List[Dict], retriever_docs: Dict[str,str], nli_threshold: float = 0.5) -> Dict:
        """
        For each claim - supporting snippet pair, run NLI (premise=snippet, hypothesis=claim).
        If model returns 'contradiction' with prob >= nli_threshold, mark as contradiction.
        Return contradiction_rate = contradicted_pairs / total_pairs.
        Also compute per-claim status (any contradiction -> claim flagged).
        """
        total_pairs = 0
        contradicted_pairs = 0
        per_claim = []
        for ci, item in enumerate(claims):
            claim_text = item.get("claim", "")
            citations = item.get("citations", [])
            claim_contradicted = False
            claim_pairs = []
            for c in citations:
                total_pairs += 1
                doc_id = c.get("doc_id") if isinstance(c, dict) else c[0]
                snippet = c.get("snippet") if isinstance(c, dict) else c[1]
                # Run NLI: premise=snippet, hypothesis=claim_text
                try:
                    out = self.nli(f"{snippet}", f"{claim_text}")  # some pipelines accept (premise, hypothesis) as tuple, but huggingface pipeline text-classification expects single input; use model's default reasoning by joining
                    # When return_all_scores=True, out is list of label-score dicts
                    # out example: [[{'label': 'CONTRADICTION', 'score': 0.01}, ...]]
                    scores = out[0] if isinstance(out, list) and isinstance(out[0], list) else out
                    # find contradiction score - labels vary by model; normalize matching
                    contr_score = 0.0
                    entail_score = 0.0
                    for o in scores:
                        lab = o.get("label", "").upper()
                        sc = float(o.get("score", 0.0))
                        if "CONTRADI" in lab or lab.startswith("CONTRAD"):
                            contr_score = max(contr_score, sc)
                        if "ENTAIL" in lab:
                            entail_score = max(entail_score, sc)
                    is_contradiction = contr_score >= nli_threshold and contr_score > entail_score
                except Exception:
                    # fallback: if pipeline with two-arg call fails, try single string "premise [SEP] hypothesis"
                    try:
                        combo = f"premise: {snippet} hypothesis: {claim_text}"
                        out = self.nli(combo)
                        scores = out[0] if isinstance(out, list) and isinstance(out[0], list) else out
                        contr_score = 0.0
                        entail_score = 0.0
                        for o in scores:
                            lab = o.get("label", "").upper()
                            sc = float(o.get("score", 0.0))
                            if "CONTRADI" in lab or lab.startswith("CONTRAD"):
                                contr_score = max(contr_score, sc)
                            if "ENTAIL" in lab:
                                entail_score = max(entail_score, sc)
                        is_contradiction = contr_score >= nli_threshold and contr_score > entail_score
                    except Exception:
                        is_contradiction = False
                if is_contradiction:
                    contradicted_pairs += 1
                    claim_contradicted = True
                claim_pairs.append({"doc_id": doc_id, "snippet": snippet, "contradiction": is_contradiction, "contradiction_score": contr_score})
            per_claim.append({"claim_id": item.get("claim_id", ci+1), "claim": item.get("claim", ""), "contradicted": claim_contradicted, "pairs": claim_pairs})
        contradiction_rate = contradicted_pairs / total_pairs if total_pairs > 0 else None
        return {"contradiction_rate": contradiction_rate, "total_pairs": total_pairs, "contradicted_pairs": contradicted_pairs, "per_claim": per_claim}

    def compute_temporal_consistency(self, claims: List[Dict], retriever_docs: Dict[str,str]) -> Dict:
        """
        Extract dates from claims and from cited docs. If claim contains a date and cited doc's sentences mention a different date/year, mark inconsistency.
        Heuristic-based: compares years primarily.
        """
        per_claim = []
        inconsistencies = 0
        total_checked = 0
        for ci, item in enumerate(claims):
            claim_text = item.get("claim", "")
            claim_dates = extract_dates(claim_text)
            citations = item.get("citations", [])
            claim_issue = False
            details = []
            if claim_dates:
                # parse claim dates to years when possible
                parsed_claim_yrs = set()
                for cd in claim_dates:
                    pd = parse_date(cd)
                    if pd:
                        parsed_claim_yrs.add(pd.year)
                # check each cited snippet/doc text for dates
                for c in citations:
                    doc_id = c.get("doc_id") if isinstance(c, dict) else c[0]
                    snippet = c.get("snippet") if isinstance(c, dict) else c[1]
                    snippet_dates = extract_dates(snippet)
                    parsed_snip_yrs = set()
                    for sd in snippet_dates:
                        ps = parse_date(sd)
                        if ps:
                            parsed_snip_yrs.add(ps.year)
                    # if both have years and disjoint -> inconsistency
                    if parsed_claim_yrs and parsed_snip_yrs:
                        total_checked += 1
                        if parsed_claim_yrs.isdisjoint(parsed_snip_yrs):
                            inconsistencies += 1
                            claim_issue = True
                            details.append({"doc_id": doc_id, "claim_dates": list(parsed_claim_yrs), "snippet_dates": list(parsed_snip_yrs), "snippet": snippet})
            per_claim.append({"claim_id": item.get("claim_id", ci+1), "claim": claim_text, "claim_dates_raw": claim_dates, "inconsistent": claim_issue, "details": details})
        temporal_inconsistency_rate = inconsistencies / total_checked if total_checked > 0 else None
        return {"temporal_inconsistency_rate": temporal_inconsistency_rate, "total_date_checks": total_checked, "inconsistencies": inconsistencies, "per_claim": per_claim}

# ---------------------------
# Main CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser("Verifier Agent")
    parser.add_argument("--claims", type=str, default="results/generated_summary.json", help="Path to generated_summary.json")
    parser.add_argument("--retriever", type=str, default="results/retriever_output.json", help="Path to retriever output JSON (optional fallback)")
    parser.add_argument("--sim-threshold", type=float, default=0.8, help="Semantic similarity threshold for factual support")
    parser.add_argument("--nli-threshold", type=float, default=0.5, help="NLI contradiction threshold")
    parser.add_argument("--sim-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--nli-model", type=str, default="facebook/bart-large-mnli")
    parser.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    parser.add_argument("--out-json", type=str, default="results/metrics.json")
    args = parser.parse_args()

    claims = load_claims(args.claims)
    retriever_docs = load_retriever_docs(args.retriever)

    verifier = Verifier(sim_model_name=args.sim_model, nli_model_name=args.nli_model, device=args.device)

    print("Computing factual precision...")
    factual_res = verifier.compute_factual_precision(claims, retriever_docs, sim_threshold=args.sim_threshold)

    print("Computing contradiction rate via NLI...")
    contradiction_res = verifier.compute_contradiction_rate(claims, retriever_docs, nli_threshold=args.nli_threshold)

    print("Computing temporal consistency...")
    temporal_res = verifier.compute_temporal_consistency(claims, retriever_docs)

    # Aggregate into metrics
    metrics = {
        "factual_precision": factual_res.get("factual_precision"),
        "factual_total_snippets": factual_res.get("total_snippets"),
        "factual_supported_snippets": factual_res.get("supported_snippets"),
        "contradiction_rate": contradiction_res.get("contradiction_rate"),
        "contradicted_pairs": contradiction_res.get("contradicted_pairs"),
        "contradiction_total_pairs": contradiction_res.get("total_pairs"),
        "temporal_inconsistency_rate": temporal_res.get("temporal_inconsistency_rate"),
        "temporal_total_checks": temporal_res.get("total_date_checks"),
        # include per-claim details for debugging
        "per_claim": []
    }

    # Merge per-claim details
    for fc in factual_res.get("per_claim", []):
        cid = fc["claim_id"]
        # find corresponding entries
        cr = next((c for c in contradiction_res.get("per_claim", []) if c["claim_id"]==cid), {})
        tr = next((c for c in temporal_res.get("per_claim", []) if c["claim_id"]==cid), {})
        metrics["per_claim"].append({
            "claim_id": cid,
            "claim": fc.get("claim"),
            "supported_any_snippet": fc.get("supported_any"),
            "snippet_count": fc.get("snippet_count"),
            "contradicted": cr.get("contradicted"),
            "temporal_inconsistent": tr.get("inconsistent"),
            "factual_details": fc.get("details"),
            "contradiction_details": cr.get("pairs"),
            "temporal_details": tr.get("details")
        })

    # Save outputs
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # also write text summary
    txt_path = os.path.splitext(args.out_json)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("VERIFIER METRICS\n\n")
        f.write(f"Factual precision (>= {args.sim_threshold}): {metrics['factual_precision']}\n")
        f.write(f"Total cited snippets: {metrics['factual_total_snippets']}\n")
        f.write(f"Supported snippets: {metrics['factual_supported_snippets']}\n\n")
        f.write(f"Contradiction rate (NLI threshold={args.nli_threshold}): {metrics['contradiction_rate']}\n")
        f.write(f"Contradicted pairs: {metrics['contradicted_pairs']} / {metrics['contradiction_total_pairs']}\n\n")
        f.write(f"Temporal inconsistency rate: {metrics['temporal_inconsistency_rate']}\n")
        f.write(f"Date checks performed: {metrics['temporal_total_checks']}\n\n")
        f.write("Per-claim summary (claim_id: supported_any_snippet, contradicted, temporal_inconsistent)\n")
        for pc in metrics["per_claim"]:
            f.write(f"{pc['claim_id']}: {pc['supported_any_snippet']}, {pc['contradicted']}, {pc['temporal_inconsistent']}\n")

    print("Saved metrics JSON ->", args.out_json)
    print("Saved text summary ->", txt_path)

if __name__ == "__main__":
    main()
