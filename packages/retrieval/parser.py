"""
Document parser: PDF → structured pages with text, tables, and section hints.

Primary path: pdfplumber (handles most born-digital PDFs well, including tables).
OCR fallback:  pdf2image + pytesseract for pages with insufficient text.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Heading patterns — ordered from most to least specific
_HEADING_PATTERNS = [
    # Numbered: 1  /  1.2  /  1.2.3  followed by title text
    re.compile(r"^(\d+(?:\.\d+){0,3})\s{1,4}([A-Z][^\n]{2,80})$"),
    # ALL-CAPS short line (likely a section heading)
    re.compile(r"^([A-Z][A-Z\s\-/&]{3,79})$"),
    # Title-Case short line (≤ 80 chars, no terminal punctuation)
    re.compile(r"^([A-Z][a-zA-Z\s\-/&:]{3,79})$"),
]


@dataclass
class ParsedPage:
    page_number: int        # 1-based
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)
    via_ocr: bool = False


@dataclass
class ParsedDocument:
    pages: list[ParsedPage]
    checksum: str
    page_count: int

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


def _checksum(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _needs_ocr(text: str, threshold: int) -> bool:
    return len(text.strip()) < threshold


def _ocr_page(pdf_path: Path, page_number: int) -> str:
    """Render page to image and run Tesseract. Returns extracted text."""
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError as e:
        logger.warning("OCR dependencies not installed (%s); skipping OCR for page %d", e, page_number)
        return ""

    images = convert_from_path(
        str(pdf_path),
        first_page=page_number,
        last_page=page_number,
        dpi=300,
    )
    if not images:
        return ""
    return pytesseract.image_to_string(images[0], lang="eng")


def _extract_tables(page) -> list[list[list[str]]]:
    """Extract tables from a pdfplumber page object."""
    tables = []
    try:
        for table in page.extract_tables():
            cleaned = [
                [cell.strip() if isinstance(cell, str) else "" for cell in row]
                for row in table
                if any(cell for cell in row)
            ]
            if cleaned:
                tables.append(cleaned)
    except Exception as exc:
        logger.debug("Table extraction failed on page: %s", exc)
    return tables


def _table_to_text(table: list[list[str]]) -> str:
    """Convert a table (list of rows) to a plain-text representation."""
    if not table:
        return ""
    col_widths = [
        max(len(row[i]) for row in table if i < len(row))
        for i in range(max(len(row) for row in table))
    ]
    lines = []
    for row in table:
        cells = [str(row[i]) if i < len(row) else "" for i in range(len(col_widths))]
        lines.append("  ".join(c.ljust(w) for c, w in zip(cells, col_widths)))
    return "\n".join(lines)


def parse_pdf(path: Path, ocr_threshold: int = 50) -> ParsedDocument:
    """
    Parse a PDF file into structured pages.

    Args:
        path: Absolute path to the PDF file.
        ocr_threshold: Pages with fewer than this many characters trigger OCR.

    Returns:
        ParsedDocument with per-page text and tables.
    """
    try:
        import pdfplumber
    except ImportError as e:
        raise RuntimeError("pdfplumber is required for PDF parsing") from e

    checksum = _checksum(path)
    pages: list[ParsedPage] = []

    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            via_ocr = False
            text = page.extract_text() or ""

            if _needs_ocr(text, ocr_threshold):
                logger.info("Page %d has sparse text (%d chars) — attempting OCR", i, len(text))
                ocr_text = _ocr_page(path, i)
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    via_ocr = True

            tables = _extract_tables(page)
            # Append table text after the page text so it is included in chunks
            if tables:
                table_texts = "\n\n".join(_table_to_text(t) for t in tables)
                text = f"{text}\n\n[TABLE]\n{table_texts}\n[/TABLE]"

            pages.append(ParsedPage(
                page_number=i,
                text=text.strip(),
                tables=tables,
                via_ocr=via_ocr,
            ))

    return ParsedDocument(pages=pages, checksum=checksum, page_count=len(pages))


def detect_headings(text: str) -> list[tuple[int, str, str]]:
    """
    Scan text for heading lines.

    Returns:
        List of (line_index, heading_level, heading_text) where heading_level
        is "h1" | "h2" | "h3" based on pattern matched.
    """
    results = []
    for line_idx, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line or len(line) > 120:
            continue
        # Numbered heading
        m = _HEADING_PATTERNS[0].match(line)
        if m:
            depth = m.group(1).count(".")
            level = ["h1", "h2", "h3"][min(depth, 2)]
            results.append((line_idx, level, line))
            continue
        # ALL-CAPS
        if _HEADING_PATTERNS[1].match(line) and line == line.upper():
            results.append((line_idx, "h1", line))
            continue
    return results
