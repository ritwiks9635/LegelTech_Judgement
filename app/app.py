import argparse
import os
from dotenv import load_dotenv

from parser import parse_pdf
from chunker import build_chunks
from embeddings import embed_chunks, embed_text
from vector_store import (
    get_qdrant_client,
    create_collection,
    upsert_chunks,
    vector_search,
)
from bm25_search import BM25Index
from fusion_search import FusionSearch
from db_mongo import get_mongo_client, save_metadata


# ============================================================
# ENV
# ============================================================
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_pipeline(pdf_path: str, query: str, top_k: int = 5):
    print("🚀 Starting LegalTech Pipeline")

    # ---------------------------
    # MongoDB
    # ---------------------------
    db = get_mongo_client(MONGO_URI)

    # ---------------------------
    # Qdrant
    # ---------------------------
    qdrant = get_qdrant_client("qdrant_db")
    create_collection(qdrant, "legal_chunks")

    # ---------------------------
    # Parse PDF
    # ---------------------------
    print("📄 Parsing PDF...")
    metadata = parse_pdf(pdf_path, GEMINI_API_KEY)

    save_metadata(db, "judgments", metadata)

    # ---------------------------
    # Chunking
    # ---------------------------
    print("✂️ Chunking paragraphs...")
    chunks = build_chunks(metadata)

    # ---------------------------
    # Embeddings
    # ---------------------------
    print("🧠 Generating embeddings...")
    enriched_chunks = embed_chunks(chunks)

    # ---------------------------
    # Qdrant upsert
    # ---------------------------
    print("📦 Storing vectors...")
    upsert_chunks(qdrant, "legal_chunks", enriched_chunks)

    # ---------------------------
    # BM25
    # ---------------------------
    print("🔍 Building BM25 index...")
    bm25 = BM25Index()
    bm25.build(enriched_chunks)

    # ---------------------------
    # Search
    # ---------------------------
    print(f"❓ Query: {query}")
    q_vec = embed_text(query)

    fusion = FusionSearch(
        bm25_index=bm25,
        qdrant_client=qdrant,
        collection_name="legal_chunks",
    )

    results = fusion.search(query, q_vec, top_k=top_k)

    print("\n✅ RESULTS\n" + "-" * 40)
    for r in results:
        print(f"\nScore: {r['score']:.4f}")
        print(f"Chunk ID: {r['chunk_id']}")
        print(r["text_preview"])
        print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legal Judgment Pipeline CLI")
    parser.add_argument("--pdf", required=True, help="Path to judgment PDF")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--top_k", type=int, default=5)

    args = parser.parse_args()

    run_pipeline(args.pdf, args.query, args.top_k)