"""
Section-aware chunker — production-grade rewrite.

Key improvements over v1
------------------------
* ChunkingConfig dataclass — all parameters in one place.
* Sentence-boundary splits — chunks end at `. `, `? `, `! ` or `\\n\\n`
  rather than mid-word token boundaries.
* Table-block protection — text between [TABLE]…[/TABLE] markers is kept
  as an atomic unit and never split mid-row.
* Minimum chunk size — undersized fragments are merged into the next chunk
  instead of producing noise embeddings.
* Overlap uses the trailing tokens of the *decoded* previous chunk so the
  overlap text is always coherent prose.
* Heading exclusion list prevents short ALL-CAPS words (NOTE, TABLE, etc.)
  from being misclassified as section headings.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .parser import ParsedPage

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ChunkingConfig:
    """Configuration for document chunk generation.

    Parameters
    ----------
    target_tokens
        Preferred token count for a prose chunk before overlap is applied.
    overlap_tokens
        Number of trailing tokens repeated into the next chunk.
    min_tokens
        Minimum prose chunk size before undersized fragments are merged forward.
    max_table_tokens
        Maximum approximate size allowed for a single table chunk.
    sentence_boundary
        Whether chunks should prefer sentence or paragraph boundaries over raw token cuts.
    """
    target_tokens: int = 512     # ideal chunk size
    overlap_tokens: int = 64     # token overlap between adjacent chunks
    min_tokens: int = 50         # chunks below this are merged into the next
    max_table_tokens: int = 768  # tables larger than this get truncated
    sentence_boundary: bool = True


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """Chunked document segment produced by the retrieval pipeline.

    Parameters
    ----------
    chunk_index
        Zero-based index of the chunk within the document.
    text
        Chunk text payload.
    token_count
        Approximate token count for the chunk.
    section_title
        Closest heading title associated with the chunk, if any.
    heading_path
        Hierarchical heading path from outermost to innermost section.
    page_start
        First source page contributing text to the chunk.
    page_end
        Last source page contributing text to the chunk.
    source_hash
        Stable hash of the chunk text for traceability and deduplication.
    is_table_chunk
        Indicates whether the chunk originated from a table block.
    """
    chunk_index: int
    text: str
    token_count: int
    section_title: str | None
    heading_path: list[str]
    page_start: int | None
    page_end: int | None
    source_hash: str
    is_table_chunk: bool = False


# ── Internal block model ──────────────────────────────────────────────────────

@dataclass
class _Block:
    """Atomic unit of text — either prose or a full table."""
    text: str
    is_table: bool = False


@dataclass
class _Section:
    title: str | None
    heading_path: list[str]
    page_start: int | None
    page_end: int | None
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return _NEWLINE_COLLAPSE.sub("\n\n", "\n".join(self.lines)).strip()


# ── Regex constants ───────────────────────────────────────────────────────────

_NEWLINE_COLLAPSE = re.compile(r"\n{3,}")
_TABLE_BLOCK = re.compile(r"\[TABLE\](.*?)\[/TABLE\]", re.DOTALL)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\'])")
_DOUBLE_NEWLINE = re.compile(r"\n\n+")

# Heading regex — numbered sections or ALL-CAPS lines
_HEADING_RE = re.compile(
    r"^(?:(\d+(?:\.\d+){0,3})\s{1,4}[A-Z].{2,79}|[A-Z][A-Z\s\-/&]{4,79})$"
)

# Short standalone words that look ALL-CAPS but are NOT section headings
_HEADING_EXCLUSIONS = frozenset({
    "NOTE", "NOTES", "WARNING", "CAUTION", "IMPORTANT", "FIGURE",
    "TABLE", "APPENDIX", "EXAMPLE", "EXAMPLES", "SEE ALSO", "REFERENCE",
    "REFERENCES", "TBD", "N/A", "NA",
})


# ── Tokenizer helpers ─────────────────────────────────────────────────────────

def _get_tokenizer():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        logger.debug("Using tiktoken cl100k_base tokenizer")
        return enc
    except ImportError:
        logger.warning("tiktoken not installed — using word-count token estimate")
        return None


def _count_tokens(text: str, tok) -> int:
    if tok is not None:
        return len(tok.encode(text))
    return int(len(text.split()) * 1.3)


def _encode(text: str, tok) -> list[int]:
    if tok is not None:
        return tok.encode(text)
    return text.split()  # words as fallback tokens


def _decode(tokens: list, tok) -> str:
    if tok is not None:
        return tok.decode(tokens)
    return " ".join(tokens)


# ── Sentence-boundary helper ──────────────────────────────────────────────────

def _last_sentence_end(text: str) -> int:
    """
    Return the character position of the last sentence boundary in *text*,
    searching from 70% of len(text) to the end.

    Falls back to the last double-newline, then to len(text).
    """
    window_start = int(len(text) * 0.70)
    window = text[window_start:]

    # Sentence-ending punctuation followed by whitespace + capital
    matches = list(_SENTENCE_END.finditer(window))
    if matches:
        return window_start + matches[-1].end()

    # Paragraph break
    matches = list(_DOUBLE_NEWLINE.finditer(window))
    if matches:
        return window_start + matches[-1].end()

    return len(text)


# ── Block splitting ───────────────────────────────────────────────────────────

def _to_blocks(text: str) -> list[_Block]:
    """
    Split section text into alternating prose blocks and table blocks.
    Table blocks are [TABLE]…[/TABLE] spans emitted by the parser.
    """
    blocks: list[_Block] = []
    last = 0
    for m in _TABLE_BLOCK.finditer(text):
        if m.start() > last:
            prose = text[last : m.start()].strip()
            if prose:
                blocks.append(_Block(text=prose, is_table=False))
        blocks.append(_Block(text=m.group(0).strip(), is_table=True))
        last = m.end()
    if last < len(text):
        prose = text[last:].strip()
        if prose:
            blocks.append(_Block(text=prose, is_table=False))
    return blocks


def _split_prose_block(text: str, cfg: ChunkingConfig, tok) -> list[str]:
    """
    Split a prose block into overlapping token-bounded chunks, ending at
    sentence boundaries when possible.
    """
    tokens = _encode(text, tok)
    if not tokens:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(tokens):
        end = min(start + cfg.target_tokens, len(tokens))
        chunk_text = _decode(tokens[start:end], tok)

        # Try to end at a sentence boundary (only if not the last chunk)
        if cfg.sentence_boundary and end < len(tokens):
            boundary = _last_sentence_end(chunk_text)
            if 0 < boundary < len(chunk_text):
                chunk_text = chunk_text[:boundary]
                # Re-encode the trimmed text to get an accurate new end position
                trimmed_tokens = _encode(chunk_text, tok)
                end = start + len(trimmed_tokens)

        chunks.append(chunk_text.strip())

        if end >= len(tokens):
            break

        # Overlap: step back by overlap_tokens from the new end
        start = max(start + 1, end - cfg.overlap_tokens)

    return [c for c in chunks if c]


# ── Greedy block packer ───────────────────────────────────────────────────────

def _pack_blocks(
    blocks: list[_Block],
    cfg: ChunkingConfig,
    tok,
    section: _Section,
    start_idx: int,
) -> list[TextChunk]:
    """
    Greedily pack blocks into chunks respecting target_tokens.

    - Prose blocks are pre-split into sub-chunks by _split_prose_block.
    - Table blocks are kept atomic; if larger than max_table_tokens they are
      truncated with a trailing note.
    - Sub-chunks below min_tokens are merged forward into the next sub-chunk.
    """
    # Decompose all blocks into candidate "units"
    units: list[tuple[str, bool]] = []  # (text, is_table)

    for block in blocks:
        if block.is_table:
            table_text = block.text
            table_tok = _count_tokens(table_text, tok)
            if table_tok > cfg.max_table_tokens:
                # Truncate: keep first max_table_tokens worth of characters
                # (approximate — full re-encoding would be expensive)
                ratio = cfg.max_table_tokens / table_tok
                cutoff = int(len(table_text) * ratio)
                table_text = table_text[:cutoff].rstrip() + "\n… [TABLE TRUNCATED]"
                logger.debug("Table block truncated from %d to ~%d tokens", table_tok, cfg.max_table_tokens)
            units.append((table_text, True))
        else:
            sub_chunks = _split_prose_block(block.text, cfg, tok)
            units.extend((t, False) for t in sub_chunks)

    # Merge undersized units forward
    merged: list[tuple[str, bool]] = []
    pending = ""
    pending_is_table = False

    for text, is_table in units:
        tok_count = _count_tokens(text, tok)

        if pending and is_table:
            # Can't merge prose pending into a table unit — flush pending first
            if _count_tokens(pending, tok) >= cfg.min_tokens:
                merged.append((pending, pending_is_table))
            else:
                # Still too small — prepend to the next non-table unit
                text = (pending + "\n\n" + text) if not is_table else text
            pending = ""

        if tok_count < cfg.min_tokens and not is_table:
            pending = (pending + " " + text).strip() if pending else text
            pending_is_table = False
        else:
            if pending:
                text = (pending + "\n\n" + text).strip()
                pending = ""
            merged.append((text, is_table))

    if pending:
        if merged:
            last_text, last_is_table = merged[-1]
            merged[-1] = (last_text + "\n\n" + pending, last_is_table)
        else:
            merged.append((pending, pending_is_table))

    # Build TextChunk objects
    chunks: list[TextChunk] = []
    for text, is_table in merged:
        text = text.strip()
        if not text:
            continue
        chunks.append(TextChunk(
            chunk_index=start_idx + len(chunks),
            text=text,
            token_count=_count_tokens(text, tok),
            section_title=section.title,
            heading_path=list(section.heading_path),
            page_start=section.page_start,
            page_end=section.page_end,
            source_hash=_source_hash(text),
            is_table_chunk=is_table,
        ))

    return chunks


# ── Section splitting ─────────────────────────────────────────────────────────

def _heading_depth(heading: str) -> int:
    m = re.match(r"^(\d+(?:\.\d+)*)", heading)
    return m.group(1).count(".") if m else 0


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    if stripped.upper() in _HEADING_EXCLUSIONS:
        return False
    return bool(_HEADING_RE.match(stripped))


def _split_into_sections(pages: list[ParsedPage]) -> list[_Section]:
    """Walk pages and split into sections at heading lines."""
    sections: list[_Section] = []
    heading_stack: list[tuple[int, str]] = []
    current = _Section(title=None, heading_path=[], page_start=None, page_end=None)

    for page in pages:
        pnum = page.page_number
        if current.page_start is None:
            current.page_start = pnum

        for line in page.text.splitlines():
            if _is_heading(line):
                if any(l.strip() for l in current.lines):
                    current.page_end = pnum
                    sections.append(current)

                heading = line.strip()
                depth = _heading_depth(heading)
                heading_stack = [(d, h) for d, h in heading_stack if d < depth]
                heading_stack.append((depth, heading))

                current = _Section(
                    title=heading,
                    heading_path=[h for _, h in heading_stack],
                    page_start=pnum,
                    page_end=None,
                )
            else:
                current.lines.append(line)

        current.page_end = pnum

    if any(l.strip() for l in current.lines):
        sections.append(current)

    return sections


# ── Public API ────────────────────────────────────────────────────────────────

def _source_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def chunk_document(
    pages: list[ParsedPage],
    config: ChunkingConfig | None = None,
) -> list[TextChunk]:
    """Split parsed pages into retrieval-ready text chunks.

    Parameters
    ----------
    pages
        Parsed pages from :func:`packages.retrieval.parser.parse_pdf`.
    config
        Chunking parameters. Defaults to :class:`ChunkingConfig` values when omitted.

    Returns
    -------
    list[TextChunk]
        Chunked document segments preserving page and heading metadata.
    """
    cfg = config or ChunkingConfig()
    tok = _get_tokenizer()
    sections = _split_into_sections(pages)

    all_chunks: list[TextChunk] = []
    for section in sections:
        if not section.text:
            continue
        blocks = _to_blocks(section.text)
        section_chunks = _pack_blocks(blocks, cfg, tok, section, len(all_chunks))
        all_chunks.extend(section_chunks)

    # Re-index after merging across sections
    for i, chunk in enumerate(all_chunks):
        chunk.chunk_index = i

    logger.info(
        "Chunked %d pages → %d sections → %d chunks (tables: %d)",
        len(pages),
        len(sections),
        len(all_chunks),
        sum(1 for c in all_chunks if c.is_table_chunk),
    )
    return all_chunks
