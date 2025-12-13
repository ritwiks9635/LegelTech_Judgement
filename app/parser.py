import re
import json
import google.generativeai as genai
import pdfplumber
from typing import Dict, Any

try:

    import pymupdf4llm
except:
    pymupdf4llm = None

# ============================================================
# 1. Extract raw text with pymupdf4llm (best option)
# ============================================================
def extract_text_pymupdf4llm(path: str) -> str:
    if pymupdf4llm:
        try:
            return pymupdf4llm.to_markdown(path)
        except:
            pass
    return ""



def extract_text_pdfplumber(path: str) -> str:
    try:
        with pdfplumber.open(path) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages)
    except:
        return ""


def load_text(path: str) -> str:
    text = extract_text_pymupdf4llm(path)
    if text.strip():
        return text
    return extract_text_pdfplumber(path)


# ============================================================
# 2. Page-level cleaning
# ============================================================
CLEAN_PATTERNS = [
    r"Indian Kanoon.*",
    r"Signature Not Verified.*",
    r"Digitally Signed.*",
    r"Signing Date.*",
    r"Page \d+ of \d+",
    r"^\s*\d{1,2}:\d{2}:\d{2}\s*$",
    r"W\.P\.\(C\).+connected matters",
    r"\bAdvocate[s]?\b.*",
]


def clean_page(text: str) -> str:
    for pat in CLEAN_PATTERNS:
        text = re.sub(pat, "", text, flags=re.I | re.M)
    return text.strip()


# ============================================================
# 3. Detect judgment start
# ============================================================
def detect_judgment_start(pages: list) -> int:
    for i, pg in enumerate(pages):
        if re.search(r"\bJUDGMENT\b|\bORDER\b|Per [A-Za-z ]+ J|1\.\s+The|1\.\s+In the", pg, re.I):
            return i
    return 0


# ============================================================
# 4. Extract paragraphs
# ============================================================
def extract_paragraphs(text: str) -> list:
    paras = []
    blocks = [b.strip() for b in re.split(r"\n{2,}", text) if b.strip()]
    for i, blk in enumerate(blocks):
        paras.append({"id": f"p{i+1}", "text": blk})
    return paras


# ============================================================
# 5. Extract tables
# ============================================================
def extract_tables(path: str) -> list:
    tables = []
    try:
        with pdfplumber.open(path) as pdf:
            tid = 1
            for page in pdf.pages:
                tbs = page.extract_tables()
                for t in tbs:
                    tables.append({"table_id": tid, "rows": t})
                    tid += 1
    except:
        pass
    return tables


# ============================================================
# 6. Gemini strict JSON extractor (fixed JSONDecodeError)
# ============================================================
def clean_json_output(text: str) -> Dict[str, Any]:
    text = re.sub(r'^\s*```(json)?\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON object found")

    json_str = m.group(0)

    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*\]", "]", json_str)

    return json.loads(json_str)


def gemini_extract(text: str, api_key: str) -> dict:
    genai.configure(api_key=api_key)

    prompt = f"""
Extract STRICT JSON metadata from the following cleaned legal judgment.

JSON FORMAT (MANDATORY):
{{
 "title": "",
 "court": "",
 "date": "",
 "facts": "",
 "issues": [],
 "arguments": {{
      "petitioner": "",
      "respondent": ""
 }},
 "ratio": "",
 "holding": "",
 "citations": []
}}

RULES:
- Output ONLY JSON (no explanation).
- Facts must be narrative.
- Issues must be 1-10 short legal issues.
- Arguments must be real court arguments.
- Ratio must contain reasoning.
- Holding must contain final order.
- Citations must be exact.
- No invented content.
- No markdown.
- No comments.

TEXT:
{text}
"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    resp = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "text/plain"}
    )

    return clean_json_output(resp.text)


# ============================================================
# 7. MAIN PARSER
# ============================================================
def parse_pdf(path: str, api_key: str) -> Dict[str, Any]:
    raw = load_text(path)
    pages = [clean_page(p) for p in re.split(r"\n{2,}", raw) if p.strip()]

    start = detect_judgment_start(pages)

    clean_text = "\n\n".join(pages[start:])

    meta = gemini_extract(clean_text, api_key)

    if not meta.get("court") or meta["court"].strip() == "":
        court_match = re.search(
            r"(SUPREME COURT OF INDIA|HIGH COURT OF [A-Z][A-Za-z]+|IN THE SUPREME COURT.*|IN THE HIGH COURT OF .*)",
            clean_text,
            re.I
        )
        meta["court"] = court_match.group(0).strip() if court_match else "Unknown Court"

    # -----------------------------------------------------
    # FIX: Guarantee Ratio & Holding always filled (using fallback logic)
    # -----------------------------------------------------
    if not meta.get("ratio") or len(meta["ratio"].strip()) < 20:
        ratio_fallback = re.findall(
            r"(ratio decidendi.*?\.|reasoning.*?\.|analysis.*?\. )",
            clean_text,
            re.I | re.S
        )
        if ratio_fallback:
            meta["ratio"] = ratio_fallback[0].strip()

    if not meta.get("holding") or len(meta["holding"].strip()) < 20:
        holding_fallback = re.findall(
            r"(final order.*?\.|held that.*?\.|the court held.*?\. )",
            clean_text,
            re.I | re.S
        )
        if holding_fallback:
            meta["holding"] = holding_fallback[0].strip()

    paragraphs = extract_paragraphs(clean_text)
    tables = extract_tables(path)

    meta["paragraphs"] = paragraphs
    meta["tables"] = tables

    return meta