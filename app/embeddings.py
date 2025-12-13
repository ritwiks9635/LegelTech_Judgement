from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import threading


# ============================================================
# Thread-safe Singleton Loader (prevents multiple model loads)
# ============================================================
_model = None
_model_lock = threading.Lock()


def load_embedding_model() -> SentenceTransformer:
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer("all-MiniLM-L6-v2")

    return _model


# ============================================================
# Generate embedding for single text
# ============================================================
def embed_text(text: str) -> List[float]:
    model = load_embedding_model()
    return model.encode(text, normalize_embeddings=True).tolist()


# ============================================================
# Generate embeddings for list of chunk dicts
# ============================================================
def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Input: List of chunk dicts from chunker.py
    Output: List of dicts with added 'embedding'
    """

    model = load_embedding_model()

    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, normalize_embeddings=True)

    enriched = []

    for chunk, vector in zip(chunks, vectors):
        enriched.append({
            **chunk,
            "embedding": vector.tolist()
        })

    return enriched
