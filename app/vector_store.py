from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
)


# ============================================================
# Qdrant Client Initialization
# ============================================================
def get_qdrant_client(path: str = "qdrant_db"):
    """
    Local, persistent Qdrant instance.
    Creates folder automatically.
    """
    return QdrantClient(path=path)


# ============================================================
# Create or Reset Collection
# ============================================================
def create_collection(
    client: QdrantClient,
    collection_name: str = "legal_chunks",
    vector_dim: int = 384,
):
    if client.collection_exists(collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_dim,
            distance=Distance.COSINE
        )
    )


# ============================================================
# Insert chunk embeddings into Qdrant
# ============================================================
def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    enriched_chunks: List[Dict[str, Any]]
):
    """
    enriched_chunks come from embed_chunks() and contain:
    - chunk_id
    - text
    - paragraph_ids
    - case_title
    - citation_count
    - section
    - embedding (vector)
    """

    points = []

    for idx, c in enumerate(enriched_chunks):
        points.append(
            PointStruct(
                id=idx + 1,               
                vector=c["embedding"],    
                payload={
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "case_title": c["case_title"],
                    "citation_count": c["citation_count"],
                    "paragraph_ids": c["paragraph_ids"],
                    "section": c["section"],
                }
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points
    )


# ============================================================
# Semantic Search
# ============================================================
def vector_search(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    top_k: int = 5
):
    """
    Returns: list of result payloads with scores.
    Works for both embedded and remote Qdrant.
    """

    result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k
    )

    final = []
    for r in result.points:
        final.append({
            "score": float(r.score),
            "chunk_id": r.payload.get("chunk_id"),
            "text_preview": r.payload.get("text")[:200],
            "section": r.payload.get("section"),
            "case_title": r.payload.get("case_title"),
            "citation_count": r.payload.get("citation_count"),
            "paragraph_ids": r.payload.get("paragraph_ids"),
        })

    return final
