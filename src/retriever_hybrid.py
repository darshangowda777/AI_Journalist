#!/usr/bin/env python3
"""
retriever_hybrid.py

Hybrid retriever (BM25 sparse + Chroma dense) for AI Journalist.

Outputs:
- results/retriever_output.json (for single query: list of docs)
- OR results/retriever_output_by_query.json (for multiple queries from data/queries.json)

Each output doc:
{
  "ID": "...",
  "title": "...",
  "text": "...",
  "source": "...",
  "date": "...",
  "sparse_score": 0.123,
  "dense_score": 0.456,
  "hybrid_score": 0.5
}
"""

#!/usr/bin/env python3
"""
retriever_hybrid.py

Hybrid retriever (BM25 sparse + Chroma dense) for AI Journalist.

Outputs:
- results/retriever_output.json (for a single sample query)
- You can also call retriever.search(...) directly for other queries.

Each output doc:
{
  "ID": "...",
  "title": "...",
  "text": "...",
  "source": "...",
  "date": "...",
  "sparse_score": 0.123,
  "dense_score": 0.456,
  "hybrid_score": 0.5
}
"""

#!/usr/bin/env python3
"""
retriever_hybrid.py

Hybrid retriever (BM25 sparse + Chroma dense) for AI Journalist.

Usage (example):
python3 retriever_hybrid.py --csv /Users/darshan/Desktop/AI_Journalist/MiniHackathon/data/news_corpus.csv \
    --query "man riding horse" --k 5 --alpha 0.5

Outputs:
- results/retriever_output.json
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb


def load_corpus_from_csv(csv_path: str, text_cols_priority=None, id_col_candidates=None):
    """
    Load CSV and produce documents (strings) and metadata dicts.
    text_cols_priority: list of column names to use for text (tries in order)
    id_col_candidates: list of ID column names to try
    """
    if text_cols_priority is None:
        text_cols_priority = ["text", "content", "body", "article", "title"]

    if id_col_candidates is None:
        id_col_candidates = ["ID", "id", "doc_id", "docID"]

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Determine which column we'll use as 'title' if available
    title_col = "title" if "title" in df.columns else None

    # Determine which column to use as text/body
    text_col = None
    for c in text_cols_priority:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        text_col = df.columns[0]

    # Determine ID column
    id_col = None
    for c in id_col_candidates:
        if c in df.columns:
            id_col = c
            break

    documents = []
    meta = []
    for idx, row in df.iterrows():
        title = str(row[title_col]) if title_col and title_col in row and pd.notna(row[title_col]) else ""
        body = str(row[text_col]) if text_col in row and pd.notna(row[text_col]) else ""
        if title and body and title != body:
            full_text = f"{title}. {body}"
        else:
            full_text = title or body or ""

        documents.append(full_text)

        meta_dict = {
            "ID": str(row[id_col]) if id_col and id_col in row and pd.notna(row[id_col]) else f"doc_{idx}",
            "title": title,
            "text": body,
            "source": str(row["source"]) if "source" in row and pd.notna(row["source"]) else "",
            "date": str(row["date"]) if "date" in row and pd.notna(row["date"]) else ""
        }
        meta.append(meta_dict)

    return documents, meta


class HybridRetriever:
    """
    Hybrid retriever combining BM25 sparse + Chroma dense retrieval.
    """

    def __init__(
        self,
        documents,
        meta=None,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=4096,
        encode_batch_size=128,
    ):
        print("Initializing Hybrid Retriever...")
        self.documents = documents
        self.meta = meta if meta is not None else [{} for _ in documents]
        self.doc_ids = [f"doc_{i}" for i in range(len(documents))]

        # BM25
        tokenized_corpus = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Sentence Transformer
        print("Loading sentence-transformer model:", model_name)
        self.model = SentenceTransformer(model_name)

        # Chroma in-memory client
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(
            name="hybrid_retriever_docs",
            metadata={"hnsw:space": "cosine"},
        )

        # Clear existing collection if present to avoid duplicates (best-effort; not fatal if API differs)
        try:
            if getattr(self.collection, "count", None) and self.collection.count() > 0:
                # Some chroma versions let you delete; try best-effort deletion
                try:
                    self.collection.delete()
                except Exception:
                    # fallback: recreate collection by name
                    self.collection = self.chroma_client.get_or_create_collection(
                        name="hybrid_retriever_docs",
                        metadata={"hnsw:space": "cosine"},
                    )
        except Exception:
            # ignore if count/delete not available
            pass

        # Generate embeddings and add in safe batches
        self._index_embeddings_in_batches(chunk_size=chunk_size, encode_batch_size=encode_batch_size)

    def _index_embeddings_in_batches(self, chunk_size=4096, encode_batch_size=128):
        """
        Encode documents in sub-batches and add to Chroma in chunks of `chunk_size`.
        This avoids backend 'max batch size' or OOM issues.
        """
        n_docs = len(self.documents)
        print(f"Indexing {n_docs} documents in chunks (chunk_size={chunk_size}, encode_batch_size={encode_batch_size})...")
        for start in range(0, n_docs, chunk_size):
            end = min(n_docs, start + chunk_size)
            batch_docs = self.documents[start:end]
            batch_ids = self.doc_ids[start:end]

            batch_embeddings = []
            for s in range(0, len(batch_docs), encode_batch_size):
                sub_docs = batch_docs[s : s + encode_batch_size]
                sub_emb = self.model.encode(sub_docs, show_progress_bar=False)
                # Convert to list-of-lists
                if hasattr(sub_emb, "tolist"):
                    sub_emb = sub_emb.tolist()
                else:
                    sub_emb = [list(e) for e in sub_emb]
                batch_embeddings.extend(sub_emb)

            # Add this chunk to chroma
            self.collection.add(
                embeddings=batch_embeddings,
                documents=batch_docs,
                ids=batch_ids,
            )
            print(f"  Indexed docs {start}..{end-1} (count={end-start})")

        print("Indexing complete. Documents indexed:", n_docs)

    @staticmethod
    def _normalize_scores(results):
        """
        Normalize list of (idx, score) into dict idx->norm_score in [0,1] (min-max)
        """
        if not results:
            return {}
        scores = [s for _, s in results]
        min_s, max_s = min(scores), max(scores)
        if max_s - min_s == 0:
            return {idx: 1.0 for idx, _ in results}
        return {idx: (s - min_s) / (max_s - min_s) for idx, s in results}

    def search(self, query: str, k: int = 5, alpha: float = 0.5, k_multiplier: int = 10):
        """
        Perform hybrid search returning top-k items.
        alpha: weight on dense (0.0 => sparse-only, 1.0 => dense-only)
        k_multiplier: candidate pool multiplier
        """
        k_fetch = max(k * k_multiplier, k)
        # Sparse (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_sparse_idx = np.argsort(bm25_scores)[-k_fetch:][::-1]
        sparse_results = [(int(i), float(bm25_scores[i])) for i in top_sparse_idx if bm25_scores[i] > 0]

        # Dense (Chroma)
        query_emb = self.model.encode([query])
        dense_search_res = self.collection.query(
            query_embeddings=query_emb.tolist(),
            n_results=k_fetch,
        )

        dense_results = []
        dense_ids = dense_search_res.get("ids", [])[0] if dense_search_res.get("ids") else []
        distances = dense_search_res.get("distances", [])[0] if dense_search_res.get("distances") else []

        for _id, dist in zip(dense_ids, distances):
            try:
                idx = int(_id.split("_")[1])
            except Exception:
                idx = self.doc_ids.index(_id) if _id in self.doc_ids else None
            if idx is None:
                continue
            similarity = 1.0 - float(dist)
            dense_results.append((idx, similarity))

        # Normalize
        norm_sparse = self._normalize_scores(sparse_results)
        norm_dense = self._normalize_scores(dense_results)

        # Candidate set
        candidate_indices = set(list(norm_sparse.keys()) + list(norm_dense.keys()))

        fused = []
        for idx in candidate_indices:
            s_score = norm_sparse.get(idx, 0.0)
            d_score = norm_dense.get(idx, 0.0)
            hybrid_score = alpha * d_score + (1 - alpha) * s_score
            fused.append((idx, {"sparse_score": round(s_score, 6), "dense_score": round(d_score, 6), "hybrid_score": round(hybrid_score, 6)}))

        # Sort by hybrid_score desc
        fused_sorted = sorted(fused, key=lambda x: x[1]["hybrid_score"], reverse=True)

        # Build output
        output = []
        for idx, scores in fused_sorted[:k]:
            meta = self.meta[idx] if idx < len(self.meta) else {}
            out_doc = {
                "ID": meta.get("ID", f"doc_{idx}"),
                "title": meta.get("title", "")[:200],
                "text": meta.get("text", self.documents[idx])[:2000],
                "source": meta.get("source", ""),
                "date": meta.get("date", ""),
                "sparse_score": scores["sparse_score"],
                "dense_score": scores["dense_score"],
                "hybrid_score": scores["hybrid_score"],
            }
            output.append(out_doc)

        return output


def main():
    parser = argparse.ArgumentParser(description="Hybrid Retriever (BM25 + Chroma)")
    parser.add_argument("--csv", "-c", type=str, required=False,
                        default="/Users/darshan/Desktop/AI_Journalist/MiniHackathon/data/news_corpus.csv",
                        help="Path to news_corpus.csv")
    parser.add_argument("--query", "-q", type=str, required=False, default="man riding horse", help="Search query")
    parser.add_argument("--k", type=int, default=5, help="Number of top results to return")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for dense (0.0 sparse-only, 1.0 dense-only)")
    parser.add_argument("--chunk-size", type=int, default=4096, help="Number of docs per add() chunk to Chroma")
    parser.add_argument("--encode-batch-size", type=int, default=128, help="Sub-batch size for embedding encoding")
    args = parser.parse_args()

    csv_path = args.csv
    print("Loading corpus from CSV:", csv_path)
    docs, meta = load_corpus_from_csv(csv_path)

    # For testing you can limit dataset size by uncommenting:
    # docs, meta = docs[:5000], meta[:5000]

    retriever = HybridRetriever(
        documents=docs,
        meta=meta,
        chunk_size=args.chunk_size,
        encode_batch_size=args.encode_batch_size,
    )

    print("Searching query:", args.query)
    results = retriever.search(query=args.query, k=args.k, alpha=args.alpha)

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "retriever_output.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Saved results to", out_path)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
