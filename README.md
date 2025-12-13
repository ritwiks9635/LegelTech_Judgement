# LegalTech Judgment Search – Proof of Concept

This project is an end-to-end LegalTech proof-of-concept system that ingests raw legal judgment PDFs, converts them into a structured and searchable knowledge base, and enables natural-language querying over the content.

The system combines:
- PDF parsing and metadata extraction
- Legal-aware chunking
- Semantic vector search
- Keyword (BM25) search
- Fusion ranking
- Persistent storage using MongoDB and Qdrant
- API and CLI-based access

The goal is to demonstrate a scalable and production-ready architecture for legal document intelligence.

---

## Project Structure

```

legaltech-poc/
│
├── README.md
├── requirements.txt
│
├── app/
│   ├── main.py                 # FastAPI app (search + ingest APIs)
│   ├── app.py                  # CLI pipeline runner (end-to-end test)
│   ├── parser.py               # PDF → metadata JSON extractor
│   ├── chunker.py              # Paragraph chunking (200–400 tokens)
│   ├── embeddings.py           # SentenceTransformer embeddings
│   ├── vector_store.py         # Qdrant vector store wrapper
│   ├── bm25_search.py          # BM25 keyword search
│   ├── fusion_search.py        # BM25 + semantic fusion search
│   ├── db_mongo.py             # MongoDB persistence layer
│
├── data/
│   ├── pdfs/                   # Sample judgment PDFs
│
└── tests/
└── test_pipeline.py        # End-to-end pipeline test

````

---

## System Overview

### Ingestion Pipeline
1. Upload or provide a legal judgment PDF
2. Parse the document into structured metadata:
   - Title
   - Court
   - Date
   - Facts
   - Issues
   - Arguments
   - Ratio (if present)
   - Holding
   - Citations
   - Paragraphs
3. Store parsed metadata in MongoDB
4. Chunk paragraphs into 200–400 token segments
5. Generate sentence embeddings
6. Store vectors in Qdrant
7. Build BM25 keyword index

### Search Pipeline
- Accepts a natural-language query
- Executes:
  - Semantic vector search
  - BM25 keyword search
- Fuses both result sets into a single ranked response
- Returns the most relevant legal text chunks

---

## Requirements

- Python 3.9+
- MongoDB Atlas account (or local MongoDB)
- Local filesystem access (for Qdrant persistence)

Install dependencies:
```bash
pip install -r requirements.txt
````

---

## Environment Variables

Create a `.env` file in the project root:

```env
MONGO_URI=mongodb+srv://<username>:<password>@<cluster-url>/?retryWrites=true&w=majority
GEMINI_API_KEY=your_api_key_here
```

Ensure your MongoDB Atlas cluster:

* Is running
* Has IP whitelist set to `0.0.0.0/0` (for testing)
* Uses correct database user credentials

---

## Running the CLI Pipeline (Recommended First)

The CLI runner validates the **entire pipeline** before using the API.

```bash
python -B app/app.py --pdf data/pdfs/sample_judgment.pdf --query "What did the court hold?"
```

This will:

* Parse the PDF
* Store metadata in MongoDB
* Chunk and embed text
* Index vectors and keywords
* Execute a fused search
* Print ranked results in the terminal

---

## Running the API Server

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

### Available Endpoints

#### Ingest a PDF

```
POST /ingest
```

* Upload a legal judgment PDF
* Builds search indexes automatically

#### Search Judgments

```
GET /search?query=your question
```

* Returns ranked legal text chunks
* Uses BM25 + semantic fusion

---

## Testing

Run the end-to-end pipeline test:

```bash
python tests/test_pipeline.py
```

This verifies:

* Parsing
* Chunking
* Embeddings
* Vector storage
* Keyword search
* Fusion ranking

---

## Design Principles

* Modular, production-ready architecture
* Clear separation between ingestion, storage, and retrieval
* CLI-first testing before API exposure
* Scalable to large legal document collections
* Compatible with both local and cloud deployments

---

## Notes

* Some judgments may not explicitly contain a “ratio decidendi”; in such cases the `ratio` field may be empty by design.
* The system prioritizes robustness and correctness over aggressive assumptions about document structure.
* All components are designed to be replaceable or extensible (e.g., embedding models, vector stores).

---

## Outcome

The final system is a fully functional legal document intelligence pipeline capable of:

* Ingesting raw PDFs
* Creating a queryable legal knowledge base
* Retrieving legally meaningful answers using natural language

```
