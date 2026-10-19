"""PDF parsing and page-aware chunking."""

import hashlib
import re
from dataclasses import dataclass

CHUNK_CHARS = 1400
OVERLAP_CHARS = 200

_REFERENCES_RE = re.compile(r"^\s*(\d+\.?\s*)?(references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ParsedPdf:
    pages: list[str]
    title: str | None


@dataclass
class Chunk:
    id: str
    paper_id: str
    index: int
    page: int
    text: str


def paper_id_for(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def parse_pdf(data: bytes) -> ParsedPdf:
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        pages = [page.get_text("text") for page in doc]
        title = (doc.metadata or {}).get("title") or None
    return ParsedPdf(pages=pages, title=title)


def _clean(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_references(pages: list[str]) -> list[str]:
    """Drop the bibliography: it matches every query keyword and answers none of them."""
    for i, page in enumerate(pages):
        if i < 2:
            continue
        if match := _REFERENCES_RE.search(page):
            return pages[:i] + [page[: match.start()]]
    return pages


def _split_page(text: str) -> list[str]:
    if len(text) <= CHUNK_CHARS:
        return [text]
    sentences = _SENTENCE_END_RE.split(text)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > CHUNK_CHARS:
            pieces.append(current)
            current = current[-OVERLAP_CHARS:].split(" ", 1)[-1] + " " + sentence
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        pieces.append(current)
    return pieces


def chunk_pages(paper_id: str, pages: list[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_no, raw in enumerate(_strip_references(pages), start=1):
        text = _clean(raw)
        if len(text) < 80:
            continue
        for piece in _split_page(text):
            chunks.append(Chunk(id=f"{paper_id}:{len(chunks)}", paper_id=paper_id, index=len(chunks), page=page_no, text=piece))
    return chunks
