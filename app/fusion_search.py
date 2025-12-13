from typing import List, Dict, Any
from typing import List, Dict, Any
from .bm25_search import BM25Index
from .vector_store import vector_search
# ============================================================
# Hybrid Search (BM25 + Vector + Weighted Fusion)
# ============================================================
class FusionSearch:

    def __init__(
        self,
        bm25_index: BM25Index,
        qdrant_client,
        collection_name: str,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ):
        """
        Fusion Search Engine

        Parameters:
        - bm25_index: BM25Index object (already built)
        - qdrant_client: QdrantClient instance
        - collection_name: Qdrant collection
        - vector_weight: weight for semantic similarity
        - bm25_weight: weight for keyword match
        """

        self.bm25 = bm25_index
        self.qdrant = qdrant_client
        self.collection = collection_name

        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    # --------------------------------------------------------
    # Fusion Search
    # --------------------------------------------------------
    def search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Returns hybrid search results:
        [
            {
                "chunk_id": "...",
                "score": 0.84,
                "text_preview": "...",
                "section": "...",
                "case_title": "..."
            }
        ]
        """

        # ---------------------------
        # 1) BM25 Keyword Search
        # ---------------------------
        bm25_hits = self.bm25.search(query, top_k=top_k)

        if bm25_hits:
            max_bm25 = max(hit["score"] for hit in bm25_hits) or 1
        else:
            max_bm25 = 1

        bm25_map = {
            hit["chunk_id"]: (hit["score"] / max_bm25)
            for hit in bm25_hits
        }

        # ---------------------------
        # 2) Vector Search (Qdrant)
        # ---------------------------
        vector_hits = vector_search(
            self.qdrant,
            self.collection,
            query_embedding,
            top_k=top_k
        )

        if vector_hits:
            max_vec = max(hit["score"] for hit in vector_hits) or 1
        else:
            max_vec = 1

        vector_map = {
            hit["chunk_id"]: (hit["score"] / max_vec)
            for hit in vector_hits
        }

        # ---------------------------
        # 3) Weighted Fusion
        # ---------------------------
        combined: Dict[str, Dict[str, Any]] = {}

        for hit in bm25_hits:
            cid = hit["chunk_id"]
            combined[cid] = {
                "chunk_id": cid,
                "chunk_meta": hit,
                "bm25": bm25_map.get(cid, 0),
                "vector": 0,
            }

        for hit in vector_hits:
            cid = hit["chunk_id"]
            if cid not in combined:
                combined[cid] = {
                    "chunk_id": cid,
                    "chunk_meta": hit,
                    "bm25": 0,
                    "vector": vector_map.get(cid, 0),
                }
            else:
                combined[cid]["vector"] = vector_map.get(cid, 0)

        for cid, obj in combined.items():
            obj["final_score"] = (
                obj["bm25"] * self.bm25_weight
                + obj["vector"] * self.vector_weight
            )

        # ---------------------------
        # 4) Sort and format output
        # ---------------------------
        final_results = sorted(
            combined.values(),
            key=lambda x: x["final_score"],
            reverse=True
        )[:top_k]

        output = []
        for item in final_results:
            meta = item["chunk_meta"]

            output.append({
                "chunk_id": meta["chunk_id"],
                "score": round(item["final_score"], 4),
                "text_preview": meta["text_preview"],
                "section": meta.get("section"),
                "case_title": meta.get("case_title"),
            })

        return output
