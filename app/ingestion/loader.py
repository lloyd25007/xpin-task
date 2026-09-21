"""PDF parsing.

Docling handles the real work -- layout model, reading order, table structure --
and ``pypdf``/``pdfplumber`` stand behind it as fallbacks so a Docling failure
degrades the parse rather than failing the ingest.
"""

from __future__ import annotations

from app.core.config import settings
from langchain_core.documents import Document
from pathlib import Path
import hashlib
import json
import logging
import re


# ═══════════════════════════════════════════════════════════════════════
# Text cleaning and the pypdf / pdfplumber fallback
# ═══════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)

# A page yielding fewer characters than this is treated as a extraction failure
# worth retrying with the slower engine.
_MIN_CHARS_PER_PAGE = 120


# Typographic characters that professionally typeset PDFs are full of. Left as
# is they break naive tokenisation, bloat embeddings with near-duplicate forms,
# and crash console output on Windows' cp1252 code page.
_UNICODE_FIXUPS = {
    "\u00a0": " ", "\u2002": " ", "\u2003": " ", "\u2009": " ",  # en/em/thin spaces
    "\u200a": " ", "\u202f": " ", "\u2007": " ", "\ufeff": "",   # more spaces, BOM
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',  # smart quotes
    "\u2013": "-", "\u2014": "-", "\u2212": "-",                 # en/em dash, minus
    "\u2026": "...", "\u00ad": "",                               # ellipsis, soft hyphen
    "\u2022": "- ", "\u25cf": "- ", "\u25aa": "- ",              # bullets
}


def _clean_text(text: str) -> str:
    """Normalise whitespace and strip PDF extraction artefacts.

    Annual reports are typeset in columns, which leaves hard line breaks mid
    sentence, long runs of dot leaders in contents pages, and a spread of
    typographic Unicode (en spaces, smart quotes, en dashes). Folding those to
    ASCII equivalents means every downstream consumer -- splitter, embedder,
    LLM, and the terminal -- sees one canonical form of each character.
    """
    text = text.replace("\x00", " ")
    for bad, good in _UNICODE_FIXUPS.items():
        text = text.replace(bad, good)
    # Catch any remaining exotic Unicode separators not listed above.
    text = re.sub(r"[\u2000-\u200b\u205f\u3000]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)                  # collapse horizontal runs
    text = re.sub(r"\.{4,}", " ", text)                  # dot leaders in contents pages
    text = re.sub(r"\n{3,}", "\n\n", text)               # cap blank-line runs
    text = re.sub(r"(?<=[a-z,])\n(?=[a-z])", " ", text)  # rejoin words split across lines
    return text.strip()


def _guess_section(text: str) -> str | None:
    """Best-effort section title for a page, used as citation metadata.

    Looks at the first few short, title-like lines. This is a heuristic label
    for display only — retrieval never depends on it being right.
    """
    for line in text.splitlines()[:6]:
        line = line.strip()
        # Title-ish: short, has letters, not a bare page number or figure caption.
        if 3 < len(line) < 80 and re.search(r"[A-Za-z]", line) and not re.fullmatch(r"[\d\s.,%-]+", line):
            if line.isupper() or line.istitle():
                return line
    return None


def _extract_with_pdfplumber(pdf_path: Path, page_numbers: list[int]) -> dict[int, str]:
    """Re-extract specific pages with pdfplumber.

    Args:
        pdf_path: the PDF on disk.
        page_numbers: 1-based page numbers that pypdf handled poorly.

    Returns:
        Mapping of page number -> recovered text (may be empty if the page is
        genuinely a full-bleed image).
    """
    recovered: dict[int, str] = {}
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed; skipping fallback extraction")
        return recovered

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_no in page_numbers:
                if page_no - 1 >= len(pdf.pages):
                    continue
                try:
                    recovered[page_no] = pdf.pages[page_no - 1].extract_text() or ""
                except Exception as exc:  # one bad page must not abort the batch
                    logger.debug("pdfplumber failed on page %s: %s", page_no, exc)
    except Exception as exc:
        logger.warning("pdfplumber could not open %s: %s", pdf_path, exc)
    return recovered


def load_pdf(pdf_path: str | Path, doc_id: str | None = None, max_pages: int | None = None) -> list[Document]:
    """Load a PDF into one LangChain ``Document`` per page.

    Page-level documents are the right granularity here: they preserve accurate
    page numbers for citations, and the hierarchical splitter downstream merges
    them into parent blocks anyway.

    Args:
        pdf_path: path to the PDF.
        doc_id: stable document identifier; defaults to the filename stem.
        max_pages: stop after this many pages (smoke tests / demos).

    Returns:
        Page documents with metadata: doc_id, source, page, section.
        Pages that yielded no usable text are dropped.

    Raises:
        FileNotFoundError: if the PDF does not exist.
    """
    from pypdf import PdfReader

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc_id = doc_id or path.stem
    reader = PdfReader(str(path))
    total = len(reader.pages) if max_pages is None else min(max_pages, len(reader.pages))
    logger.info("Loading %s (%d of %d pages)", path.name, total, len(reader.pages))

    raw: dict[int, str] = {}
    weak_pages: list[int] = []

    # --- Pass 1: fast extraction with pypdf -------------------------------
    for idx in range(total):
        page_no = idx + 1
        try:
            text = reader.pages[idx].extract_text() or ""
        except Exception as exc:
            logger.debug("pypdf failed on page %s: %s", page_no, exc)
            text = ""
        raw[page_no] = text
        if len(text.strip()) < _MIN_CHARS_PER_PAGE:
            weak_pages.append(page_no)

    # --- Pass 2: recover only the pages that came back thin ---------------
    if weak_pages:
        logger.info("Retrying %d sparse pages with pdfplumber", len(weak_pages))
        for page_no, recovered in _extract_with_pdfplumber(path, weak_pages).items():
            if len(recovered.strip()) > len(raw.get(page_no, "").strip()):
                raw[page_no] = recovered

    # --- Build Documents, dropping pages that are genuinely empty ---------
    documents: list[Document] = []
    for page_no in sorted(raw):
        text = _clean_text(raw[page_no])
        if len(text) < 40:   # cover art, dividers, blank pages
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "doc_id": doc_id,
                    "source": str(path),
                    "page": page_no,
                    "section": _guess_section(text),
                },
            )
        )

    logger.info("Extracted text from %d/%d pages", len(documents), total)
    return documents

# ═══════════════════════════════════════════════════════════════════════
# Docling: layout-aware parsing with table structure
# ═══════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)


def _cache_key(pdf_path: Path, max_pages: int | None) -> str:
    """Build a cache key from the file's identity and the page limit.

    Uses size + mtime rather than a full content hash: hashing a 7MB PDF on
    every ingest costs more than it saves, and size+mtime changes whenever the
    file is actually replaced.
    """
    stat = pdf_path.stat()
    raw = f"{pdf_path.name}|{stat.st_size}|{int(stat.st_mtime)}|{max_pages}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _cache_path(key: str) -> Path:
    """Location of the cached parse for a given key."""
    cache_dir = settings.data_path / "docling_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{key}.json"


def _load_cached(key: str) -> list[Document] | None:
    """Return a cached parse, or ``None`` on a miss or unreadable cache."""
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        logger.info("Using cached Docling parse (%d pages) from %s", len(rows), path.name)
        return [Document(page_content=r["text"], metadata=r["metadata"]) for r in rows]
    except Exception as exc:
        logger.warning("Ignoring unreadable Docling cache: %s", exc)
        return None


def _save_cache(key: str, documents: list[Document]) -> None:
    """Persist a parse so the next ingest of this PDF is instant."""
    try:
        rows = [{"text": d.page_content, "metadata": d.metadata} for d in documents]
        _cache_path(key).write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:
        logger.warning("Could not write Docling cache: %s", exc)


def _build_converter():
    """Construct a Docling converter tuned for a digital, text-based PDF.

    Two settings dominate runtime and quality here:

    * ``do_ocr = False`` -- this annual report has a real text layer, so OCR
      adds nothing and costs everything. Docling enables it by default to cope
      with scanned documents; leaving it on made parsing roughly an order of
      magnitude slower on this file and produced a stream of "RapidOCR returned
      empty result" warnings, because there was nothing to recognise.
    * ``do_table_structure = True`` with cell matching -- this is the reason to
      use Docling at all. It reconstructs the table grid so a figure stays
      attached to its row label.

    Falls back to a default converter if the options API differs in the
    installed Docling version, so a version bump degrades speed rather than
    breaking ingestion.
    """
    from docling.document_converter import DocumentConverter

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption

        options = PdfPipelineOptions()
        options.do_ocr = False              # digital PDF: OCR is pure overhead
        options.do_table_structure = True   # the point of using Docling
        try:
            options.table_structure_options.do_cell_matching = True
        except AttributeError:
            pass  # older versions match cells unconditionally

        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
    except Exception as exc:
        logger.warning("Could not configure Docling pipeline (%s); using defaults", exc)
        return DocumentConverter()


def _page_of(item) -> int:
    """Read the 1-based page number off a Docling item.

    Docling records layout provenance in ``item.prov``; an item without
    provenance (rare, usually synthetic) is attributed to page 1 so it is never
    silently dropped.
    """
    prov = getattr(item, "prov", None) or []
    if prov:
        page_no = getattr(prov[0], "page_no", None)
        if isinstance(page_no, int) and page_no > 0:
            return page_no
    return 1


def _render_table(table, doc) -> str:
    """Convert a Docling table into markdown.

    Markdown is the right target: it keeps each row on one line with its header,
    which is precisely what lets the LLM associate a figure with its label. The
    export API has moved between Docling versions, so each known form is tried
    before giving up.
    """
    # Newer Docling: export_to_markdown(doc); older: export_to_markdown().
    for attempt in (lambda: table.export_to_markdown(doc), lambda: table.export_to_markdown()):
        try:
            text = attempt()
            if text and text.strip():
                return text
        except Exception:
            continue

    # Last resort: a dataframe still preserves the row/column structure.
    try:
        return table.export_to_dataframe().to_markdown(index=False)
    except Exception:
        return ""


def _convert_with_docling(pdf_path: Path, max_pages: int | None) -> list[tuple[int, str]]:
    """Run Docling and return ``(page_number, text)`` fragments in reading order.

    Text items and tables are collected separately and then interleaved by page,
    because Docling stores them in separate collections on the document.

    Raises:
        ImportError: Docling is not installed.
        Exception: conversion failed; the caller falls back to pypdf.
    """
    from docling.document_converter import DocumentConverter

    logger.info("Parsing %s with Docling (layout + table structure)", pdf_path.name)
    converter = _build_converter()

    # page_range is 1-based and inclusive; older versions lack the kwarg.
    try:
        result = (
            converter.convert(str(pdf_path), page_range=(1, max_pages))
            if max_pages
            else converter.convert(str(pdf_path))
        )
    except TypeError:
        result = converter.convert(str(pdf_path))

    doc = result.document
    fragments: list[tuple[int, str]] = []

    # --- Narrative text, already in corrected reading order --------------
    for item in getattr(doc, "texts", []) or []:
        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue
        page = _page_of(item)
        if max_pages and page > max_pages:
            continue
        fragments.append((page, text))

    # --- Tables, rendered as markdown ------------------------------------
    for table in getattr(doc, "tables", []) or []:
        page = _page_of(table)
        if max_pages and page > max_pages:
            continue
        rendered = _render_table(table, doc)
        if rendered.strip():
            fragments.append((page, f"\n{rendered}\n"))

    if not fragments:
        raise ValueError("Docling returned no text items")

    return fragments


def load_pdf_docling(pdf_path: str | Path, doc_id: str | None = None,
                     max_pages: int | None = None, use_cache: bool = True) -> list[Document]:
    """Parse a PDF with Docling into one Document per page.

    Output is deliberately identical in shape to
    :func:`app.ingestion.loader.load_pdf` -- page-level Documents carrying
    ``doc_id``, ``source``, ``page`` and ``section`` -- so the hierarchical
    chunker consumes either parser without modification.

    Args:
        pdf_path: path to the PDF.
        doc_id: stable document id; defaults to the filename stem.
        max_pages: parse only the first N pages.
        use_cache: reuse a previous parse of the same file when available.

    Returns:
        Page documents in reading order. Falls back to the pypdf/pdfplumber
        loader if Docling is unavailable or errors.

    Raises:
        FileNotFoundError: the PDF does not exist.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc_id = doc_id or path.stem
    key = _cache_key(path, max_pages)

    if use_cache:
        cached = _load_cached(key)
        if cached is not None:
            # doc_id may differ from the cached run; keep the caller's choice.
            for document in cached:
                document.metadata["doc_id"] = doc_id
            return cached

    # --- Parse, falling back to the simple extractor on any failure -------
    try:
        fragments = _convert_with_docling(path, max_pages)
    except ImportError:
        logger.warning("Docling is not installed; falling back to pypdf")
        return load_pdf_fallback(path, doc_id=doc_id, max_pages=max_pages)
    except Exception as exc:
        logger.warning("Docling parsing failed (%s); falling back to pypdf", exc)
        return load_pdf_fallback(path, doc_id=doc_id, max_pages=max_pages)

    # --- Group fragments into one Document per page -----------------------
    by_page: dict[int, list[str]] = {}
    for page, text in fragments:
        by_page.setdefault(page, []).append(text)

    documents: list[Document] = []
    for page in sorted(by_page):
        text = _clean_text("\n".join(by_page[page]))
        if len(text) < 40:  # cover art, dividers, blank pages
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "doc_id": doc_id,
                    "source": str(path),
                    "page": page,
                    "section": _guess_section(text),
                    "parser": "docling",
                },
            )
        )

    logger.info("Docling extracted %d pages from %s", len(documents), path.name)
    if use_cache and documents:
        _save_cache(key, documents)
    return documents
