import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from dotenv import load_dotenv

from .parser import parse_pdf
from .chunker import chunk_paragraphs
from .embeddings import embed_chunks, embed_text
from .vector_store import (
    get_qdrant_client,
    create_collection,
    upsert_chunks,
)
from .bm25_search import BM25Index
from .fusion_search import FusionSearch
from .db_mongo import get_mongo_client, save_metadata


# ============================================================
# ENV
# ============================================================
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not MONGO_URI or not GEMINI_API_KEY:
    raise RuntimeError("Missing environment variables")


# ============================================================
# APP INIT
# ============================================================
app = FastAPI(title="LegalTech Judgment Search API")

db = get_mongo_client(MONGO_URI)

qdrant = get_qdrant_client("qdrant_db")
create_collection(qdrant, "legal_chunks")

bm25_index = BM25Index()


# ============================================================
# INGEST
# ============================================================
@app.post("/ingest")
def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF allowed")

    pdf_path = f"/tmp/{file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(file.file.read())

    metadata = parse_pdf(pdf_path, GEMINI_API_KEY)
    save_metadata(db, "judgments", metadata)

    chunks = chunk_paragraphs(metadata["paragraphs"])

    for c in chunks:
        c["case_title"] = metadata["title"]
        c["citation_count"] = len(metadata.get("citations", []))
        c["section"] = "judgment"

    enriched = embed_chunks(chunks)
    upsert_chunks(qdrant, "legal_chunks", enriched)

    bm25_index.build(enriched)

    return {
        "status": "success",
        "title": metadata["title"],
        "chunks_indexed": len(enriched),
    }


# ============================================================
# SEARCH
# ============================================================
@app.get("/search")
def search(
    query: str = Query(..., min_length=3),
    top_k: int = 5
):
    q_vec = embed_text(query)

    fusion = FusionSearch(
        bm25_index=bm25_index,
        qdrant_client=qdrant,
        collection_name="legal_chunks",
    )

    results = fusion.search(query, q_vec, top_k)

    return {"query": query, "results": results}
