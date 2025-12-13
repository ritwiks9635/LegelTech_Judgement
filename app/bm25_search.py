from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import re


# -----------------------------------------------------------
# Lightweight, NO-ERROR tokenizer
# -----------------------------------------------------------
def tokenize(text: str) -> List[str]:
    """
    Zero-error tokenizer.
    - No NLTK
    - No downloads
    - Works for all legal text
    """
    text = text.lower()
    return re.findall(r"[a-zA-Z0-9]+", text)


# -----------------------------------------------------------
# BM25 Index — Production Safe
# -----------------------------------------------------------
class BM25Index:

    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25 = None

    # --------------------------------------------------------
    # Build BM25 Index
    # --------------------------------------------------------
    def build(self, enriched_chunks: List[Dict[str, Any]]):
        if not enriched_chunks:
            raise ValueError("BM25Index.build() received empty chunk list.")

        self.chunks = enriched_chunks

        self.tokenized_corpus = [
            tokenize(c.get("text", "")) for c in enriched_chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    # --------------------------------------------------------
    # BM25 Search
    # --------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.bm25:
            raise ValueError("BM25Index not built. Call build() first.")

        if not query or not query.strip():
            return []

        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = []
        for idx, score in ranked:
            chunk = self.chunks[idx]

            results.append({
                "chunk_id": chunk["chunk_id"],
                "score": float(score),
                "text_preview": chunk["text"][:200],
                "section": chunk.get("section"),
                "case_title": chunk.get("case_title"),
            })

        return results
