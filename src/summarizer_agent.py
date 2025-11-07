#!/usr/bin/env python3
"""
summarizer_agent.py

Task 3 — Summarizer / QA agent for AI Journalist.

Usage examples:
# using default inputs (expects results/retriever_output.json or an input JSON/CSV)
python src/summarizer_agent.py --input results/retriever_output.json --out results/generated_summary.txt --model google/flan-t5-large --max-tokens 512

# or with a smaller model (faster)
python src/summarizer_agent.py --model google/flan-t5-base --n-beams 4 --temperature 0.0
"""

#!/usr/bin/env python3
"""
summarizer_qa_agent.py

Generates a set of claims + citations from a corpus using an instruction-tuned model.
Output:
- results/generated_summary.txt  (human-readable)
- results/generated_summary.json (structured)

Usage examples:
python3 summarizer_qa_agent.py --input retriever --retriever-path results/retriever_output.json \
    --model google/flan-t5-large --num-claims 5 --top-k 10

python3 summarizer_qa_agent.py --input csv --csv-path data/news_corpus.csv \
    --model google/flan-t5-large --num-claims 7 --top-k 15
"""

#!/usr/bin/env python3
"""
summarizer_agent.py

Generates claims + citations from a corpus using an instruction-tuned model.
This version is robust to missing NLTK punkt downloads by using a fallback sentence splitter.

Outputs:
- results/generated_summary.txt
- results/generated_summary.json

Usage (example):
python3 summarizer_agent.py --input retriever --retriever-path results/retriever_output.json \
    --model google/flan-t5-large --num-claims 5 --top-k 10
"""

import os
import json
import argparse
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm

# Transformer generation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# sentence-transformers for semantic search of snippets
from sentence_transformers import SentenceTransformer, util

# ------------------------
# Robust sentence tokenizer (uses NLTK if available; otherwise a regex fallback)
# ------------------------
def try_import_nltk():
    try:
        import nltk
        from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
        return nltk, nltk_sent_tokenize
    except Exception:
        return None, None

_nltk, _nltk_sent_tokenize = try_import_nltk()

def sent_tokenize_safe(text: str) -> List[str]:
    """
    Sentence tokenizer that tries NLTK's punkt first; if unavailable, uses a regex fallback.
    The fallback is not perfect but is robust for most news-like text.
    """
    if not text:
        return []
    if _nltk_sent_tokenize is not None:
        try:
            return [s.strip() for s in _nltk_sent_tokenize(text) if s.strip()]
        except Exception:
            pass

    # Simple regex-based fallback:
    import re
    # Split on sentence-ending punctuation followed by space + capital letter, or linebreaks.
    pattern = r'(?<=[.!?])\s+(?=[A-Z0-9"\'“])|\n+'
    parts = [p.strip() for p in re.split(pattern, text) if p and p.strip()]
    # Further split very long chunks by period (best-effort)
    splitted = []
    for p in parts:
        if len(p) > 1000:
            # break long paragraph into shorter sentences by '.', '?' or '!'
            sub = [s.strip() for s in re.split(r'(?<=[.!?])\s+', p) if s.strip()]
            splitted.extend(sub)
        else:
            splitted.append(p)
    return splitted

# ---------------------
# Load docs helpers
# ---------------------
def load_docs_from_retriever(path: str, max_docs: int = None) -> Tuple[List[str], List[dict]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if max_docs:
        data = data[:max_docs]
    docs, meta = [], []
    for item in data:
        text = item.get("text", "") or item.get("title", "")
        if item.get("title") and item.get("title") not in text:
            text = f"{item.get('title')}. {text}"
        docs.append(str(text))
        meta.append({
            "ID": item.get("ID", item.get("id", "")),
            "title": item.get("title", ""),
            "source": item.get("source", ""),
            "date": item.get("date", "")
        })
    return docs, meta

def load_docs_from_csv(csv_path: str, text_col_priority=None, max_docs: int = None) -> Tuple[List[str], List[dict]]:
    if text_col_priority is None:
        text_col_priority = ["text", "content", "body", "article", "title"]
    df = pd.read_csv(csv_path)
    text_col = None
    for c in text_col_priority:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        text_col = df.columns[0]
    title_col = "title" if "title" in df.columns else None
    id_col = None
    for c in ["ID", "id", "doc_id", "docID"]:
        if c in df.columns:
            id_col = c
            break

    docs, meta = [], []
    for idx, row in df.iterrows():
        if max_docs and len(docs) >= max_docs:
            break
        body = str(row[text_col]) if pd.notna(row[text_col]) else ""
        title = str(row[title_col]) if title_col and pd.notna(row[title_col]) else ""
        text = f"{title}. {body}" if title and body and title not in body else (title or body or "")
        docs.append(str(text))
        meta.append({
            "ID": str(row[id_col]) if id_col and pd.notna(row[id_col]) else f"doc_{idx}",
            "title": title,
            "source": str(row["source"]) if "source" in row and pd.notna(row["source"]) else "",
            "date": str(row["date"]) if "date" in row and pd.notna(row["date"]) else ""
        })
    return docs, meta

# ---------------------
# Build context prompt
# ---------------------
PROMPT_TEMPLATE = """
You are an expert summarizer. Given the following context documents, produce {num_claims} concise factual claims about the content.
For each claim, provide:
1) The claim as a single short sentence.
2) A short explanation (1-2 sentences) if necessary.
3) A list of supporting citations formatted as: [DOC_ID: snippet] where DOC_ID is the document identifier and snippet is a 1-2 sentence excerpt supporting the claim.
Do not invent documents. If a claim cannot be supported by the context, mark it as "UNSUPPORTED" and do not assert it as fact.

Context documents (each doc begins with "### DOC <DOC_ID>"):
{context}

Output format (exactly):
CLAIM 1:
Claim: ...
Explanation: ...
Citations:
- [DOC_ID: snippet]
- [DOC_ID: snippet]

CLAIM 2:
...
"""

def build_context_from_docs(docs: List[str], meta: List[dict], top_k: int) -> str:
    out = []
    for i, (doc, m) in enumerate(zip(docs[:top_k], meta[:top_k])):
        doc_id = m.get("ID") or f"doc_{i}"
        doc_text = doc if len(doc) <= 2000 else doc[:2000] + "..."
        out.append(f"### DOC {doc_id}\n{doc_text}\n")
    return "\n".join(out)

# ---------------------
# Citation extraction using sentence-transformers semantic search
# ---------------------
def extract_citations_for_claims(claims: List[str], docs: List[str], meta: List[dict], top_snippets_per_claim: int = 2, sent_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    s_model = SentenceTransformer(sent_model_name)
    # split docs into sentences using safe tokenizer
    doc_sentences = []
    sent_metadata = []
    for idx, doc in enumerate(docs):
        sents = sent_tokenize_safe(doc)
        for s in sents:
            s = s.strip()
            if not s:
                continue
            doc_sentences.append(s)
            sent_metadata.append({"doc_idx": idx, "doc_id": meta[idx].get("ID", f"doc_{idx}")})

    if len(doc_sentences) == 0:
        return [[] for _ in claims]

    sent_embeddings = s_model.encode(doc_sentences, show_progress_bar=False, convert_to_tensor=True)

    claim_citations = []
    for claim in claims:
        c_emb = s_model.encode(claim, convert_to_tensor=True)
        hits = util.semantic_search(c_emb, sent_embeddings, top_k=top_snippets_per_claim)
        hits = hits[0] if isinstance(hits, list) and len(hits) > 0 else hits
        snippets = []
        for hit in hits:
            sent_idx = hit["corpus_id"]
            score = float(hit["score"])
            sent_text = doc_sentences[sent_idx]
            meta_info = sent_metadata[sent_idx]
            snippets.append((meta_info["doc_id"], sent_text, score))
        claim_citations.append(snippets)
    return claim_citations

# ---------------------
# Generation with seq2seq model (FLAN-T5 style)
# ---------------------
def generate_claims_with_model(model_name: str, context: str, num_claims: int, device: str = "cpu", max_new_tokens: int = 256):
    print("Loading generator model:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device_idx = 0 if device == "cuda" else -1
    gen_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device_idx)

    prompt = PROMPT_TEMPLATE.format(num_claims=num_claims, context=context)
    print("Generating claims (this can take a moment)...")
    # use max_new_tokens to avoid conflicting arguments warning
    out = gen_pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=False, num_return_sequences=1)[0]["generated_text"]
    return out

# ---------------------
# Parse model output for Claims
# ---------------------
def parse_generated_claims(raw_text: str) -> List[str]:
    lines = raw_text.splitlines()
    claims = []
    current_claim = None
    for line in lines:
        l = line.strip()
        if l.upper().startswith("CLAIM"):
            if current_claim:
                claims.append(current_claim.strip())
            current_claim = ""
        elif l.startswith("Claim:"):
            text = l[len("Claim:"):].strip()
            if current_claim is not None:
                current_claim = text
        elif l and current_claim is not None:
            # stop appending if we've hit Explanation or Citations headers
            if l.startswith("Explanation:") or l.startswith("Citations:") or l.startswith("- ["):
                continue
            else:
                # append possible continuation of claim
                if current_claim == "":
                    current_claim = l
                else:
                    current_claim += " " + l
    if current_claim:
        claims.append(current_claim.strip())

    # Fallback: if none extracted, split into sentences and pick first N
    if not claims:
        sents = [s.strip() for s in sent_tokenize_safe(raw_text) if s.strip()]
        return sents
    return claims

# ---------------------
# CLI & main
# ---------------------
def main():
    parser = argparse.ArgumentParser("Summarizer / QA Agent (claims + citations)")
    parser.add_argument("--input", choices=["retriever", "csv"], default="retriever")
    parser.add_argument("--retriever-path", type=str, default="results/retriever_output.json")
    parser.add_argument("--csv-path", type=str, default="data/news_corpus.csv")
    parser.add_argument("--model", type=str, default="google/flan-t5-large")
    parser.add_argument("--num-claims", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--snippets-per-claim", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", type=str, choices=["cpu","cuda"], default="cpu")
    args = parser.parse_args()

    if args.input == "retriever":
        if not os.path.isfile(args.retriever_path):
            raise FileNotFoundError(f"Retriever output not found: {args.retriever_path}")
        docs, meta = load_docs_from_retriever(args.retriever_path, max_docs=args.top_k)
    else:
        if not os.path.isfile(args.csv_path):
            raise FileNotFoundError(f"CSV not found: {args.csv_path}")
        docs, meta = load_docs_from_csv(args.csv_path, max_docs=args.top_k)

    context = build_context_from_docs(docs, meta, top_k=args.top_k)
    raw_out = generate_claims_with_model(model_name=args.model, context=context, num_claims=args.num_claims, device=args.device, max_new_tokens=args.max_new_tokens)

    claims = parse_generated_claims(raw_out)
    claims = claims[: args.num_claims]

    claim_citations = extract_citations_for_claims(claims, docs, meta, top_snippets_per_claim=args.snippets_per_claim)

    structured = []
    for i, claim in enumerate(claims):
        citations = claim_citations[i] if i < len(claim_citations) else []
        items = [{"doc_id": c[0], "snippet": c[1], "score": c[2]} for c in citations]
        structured.append({
            "claim_id": i+1,
            "claim": claim,
            "citations": items
        })

    os.makedirs("results", exist_ok=True)
    txt_path = os.path.join("results", "generated_summary.txt")
    json_path = os.path.join("results", "generated_summary.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("GENERATED CLAIMS + CITATIONS\n\n")
        for item in structured:
            f.write(f"CLAIM {item['claim_id']}:\n")
            f.write(f"Claim: {item['claim']}\n")
            if item['citations']:
                f.write("Citations:\n")
                for c in item['citations']:
                    f.write(f"- [{c['doc_id']}: {c['snippet']}] (score={c['score']:.4f})\n")
            else:
                f.write("Citations: NONE found\n")
            f.write("\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)

    print("Saved summary TXT ->", txt_path)
    print("Saved structured JSON ->", json_path)
    print("\nRaw model output:\n")
    print(raw_out)

if __name__ == "__main__":
    main()
