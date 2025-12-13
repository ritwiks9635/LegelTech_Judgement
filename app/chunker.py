import nltk
import tiktoken
from typing import List, Dict, Any


nltk.download("punkt", quiet=True)


# ============================================================
# Tokenizer Helper (OpenAI tiktoken tokenizer)
# ============================================================
def count_tokens(text: str, model: str = "gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


# ============================================================
# Safe sentence split for long paragraphs
# ============================================================
def split_paragraph_into_sentences(paragraph: str) -> List[str]:
    try:
        return nltk.sent_tokenize(paragraph)
    except:
        return [paragraph]


# ============================================================
# Main Chunker Function
# ============================================================
def build_chunks(
    metadata_json: Dict[str, Any],
    min_tokens: int = 200,
    max_tokens: int = 400,
) -> List[Dict[str, Any]]:
    """
    Convert parsed metadata paragraphs → 200–400 token chunks
    without losing legal meaning.
    """

    chunks = []
    buffer = ""
    buffer_paragraphs = []
    chunk_id = 1

    paragraphs = metadata_json.get("paragraphs", [])

    for para in paragraphs:
        text = para["text"].strip()

        if not text:
            continue

        para_tokens = count_tokens(text)

        if para_tokens > max_tokens:
            sentences = split_paragraph_into_sentences(text)

            for sentence in sentences:
                sent_tokens = count_tokens(sentence)

                if count_tokens(buffer) + sent_tokens <= max_tokens:
                    buffer += " " + sentence
                    buffer_paragraphs.append(para["id"])
                else:
                    chunks.append({
                        "chunk_id": f"chunk_{chunk_id}",
                        "text": buffer.strip(),
                        "paragraph_ids": list(set(buffer_paragraphs)),
                    })
                    chunk_id += 1

                    buffer = sentence
                    buffer_paragraphs = [para["id"]]
            continue

        if count_tokens(buffer) + para_tokens <= max_tokens:
            buffer += " " + text
            buffer_paragraphs.append(para["id"])
        else:
            if buffer.strip():
                chunks.append({
                    "chunk_id": f"chunk_{chunk_id}",
                    "text": buffer.strip(),
                    "paragraph_ids": list(set(buffer_paragraphs)),
                })
                chunk_id += 1

            buffer = text
            buffer_paragraphs = [para["id"]]

    if buffer.strip():
        chunks.append({
            "chunk_id": f"chunk_{chunk_id}",
            "text": buffer.strip(),
            "paragraph_ids": list(set(buffer_paragraphs)),
        })

    case_title = metadata_json.get("title", "Unknown Case")
    citation_count = len(metadata_json.get("citations", []))

    for c in chunks:
        c["case_title"] = case_title
        c["citation_count"] = citation_count
        c["section"] = detect_section(c["text"])

    return chunks


# ============================================================
# Section Detection (Facts / Issues / Ratio / Holding)
# ============================================================
def detect_section(text: str) -> str:
    t = text.lower()

    if "facts" in t or "background" in t or "factual" in t:
        return "facts"
    if "issue" in t or "issues" in t:
        return "issues"
    if "ratio" in t or "reasoning" in t or "analysis" in t:
        return "analysis"
    if "held" in t or "final" in t or "ordered" in t:
        return "holding"

    return "general"
