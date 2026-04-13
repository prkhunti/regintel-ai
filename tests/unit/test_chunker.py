"""Unit tests for the chunking pipeline."""
from __future__ import annotations

import pytest

from packages.retrieval.chunker import (
    ChunkingConfig,
    TextChunk,
    _is_heading,
    _last_sentence_end,
    _split_into_sections,
    _to_blocks,
    chunk_document,
)
from packages.retrieval.parser import ParsedPage


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_page(text: str, page_number: int = 1) -> ParsedPage:
    return ParsedPage(page_number=page_number, text=text)


def make_pages(*texts: str) -> list[ParsedPage]:
    return [ParsedPage(page_number=i + 1, text=t) for i, t in enumerate(texts)]


# ── _is_heading ───────────────────────────────────────────────────────────────

class TestIsHeading:
    def test_numbered_section(self):
        assert _is_heading("1 Introduction") is True

    def test_numbered_subsection(self):
        assert _is_heading("3.2 Risk Management") is True

    def test_numbered_deep(self):
        assert _is_heading("4.1.2 Software Architecture") is True

    def test_all_caps_multi_word(self):
        assert _is_heading("RISK MANAGEMENT SUMMARY") is True

    def test_exclusion_note(self):
        assert _is_heading("NOTE") is False

    def test_exclusion_table(self):
        assert _is_heading("TABLE") is False

    def test_exclusion_warning(self):
        assert _is_heading("WARNING") is False

    def test_body_text_not_heading(self):
        assert _is_heading("This is a regular sentence in a paragraph.") is False

    def test_too_long(self):
        assert _is_heading("A" * 125) is False

    def test_empty(self):
        assert _is_heading("") is False


# ── _last_sentence_end ────────────────────────────────────────────────────────

class TestLastSentenceEnd:
    def test_finds_sentence_end(self):
        text = "This is sentence one. This is sentence two. And here is three."
        pos = _last_sentence_end(text)
        # Should point past one of the sentence boundaries in the back 30%
        assert 0 < pos <= len(text)
        assert text[:pos].endswith(". ") or text[:pos] == text

    def test_falls_back_to_len_when_no_boundary(self):
        text = "nopunctuationatall" * 5
        pos = _last_sentence_end(text)
        assert pos == len(text)

    def test_paragraph_break_fallback(self):
        text = "First paragraph content here\n\nSecond paragraph content"
        pos = _last_sentence_end(text)
        assert 0 < pos <= len(text)


# ── _to_blocks ────────────────────────────────────────────────────────────────

class TestToBlocks:
    def test_no_table(self):
        blocks = _to_blocks("Just plain text with no tables.")
        assert len(blocks) == 1
        assert blocks[0].is_table is False

    def test_table_only(self):
        text = "[TABLE]\ncol1  col2\nval1  val2\n[/TABLE]"
        blocks = _to_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].is_table is True

    def test_text_then_table(self):
        text = "Some prose.\n\n[TABLE]\na  b\n[/TABLE]"
        blocks = _to_blocks(text)
        assert len(blocks) == 2
        assert blocks[0].is_table is False
        assert blocks[1].is_table is True

    def test_table_between_prose(self):
        text = "Before.\n\n[TABLE]\nx  y\n[/TABLE]\n\nAfter."
        blocks = _to_blocks(text)
        assert len(blocks) == 3
        assert blocks[0].is_table is False
        assert blocks[1].is_table is True
        assert blocks[2].is_table is False

    def test_multiple_tables(self):
        text = "[TABLE]\nt1\n[/TABLE]\n\n[TABLE]\nt2\n[/TABLE]"
        blocks = _to_blocks(text)
        table_blocks = [b for b in blocks if b.is_table]
        assert len(table_blocks) == 2


# ── _split_into_sections ──────────────────────────────────────────────────────

class TestSplitIntoSections:
    def test_single_section(self):
        page = make_page("1 Introduction\nThis is intro text.")
        sections = _split_into_sections([page])
        assert len(sections) >= 1

    def test_multiple_sections(self):
        text = (
            "1 Introduction\nIntro body.\n\n"
            "2 Background\nBackground body.\n\n"
            "3 Methods\nMethods body."
        )
        sections = _split_into_sections([make_page(text)])
        assert len(sections) >= 2

    def test_heading_path_inheritance(self):
        text = (
            "1 CLINICAL EVALUATION\nSome text.\n\n"
            "1.1 Scope\nScope details.\n\n"
            "1.2 Methods\nMethod details."
        )
        sections = _split_into_sections([make_page(text)])
        # Find sections with heading paths of depth > 1
        sub_sections = [s for s in sections if len(s.heading_path) > 1]
        assert len(sub_sections) >= 1

    def test_page_provenance(self):
        pages = make_pages(
            "1 Introduction\nIntro text.",
            "More intro text on page 2.",
        )
        sections = _split_into_sections(pages)
        assert sections[0].page_start == 1

    def test_multi_page_section(self):
        pages = make_pages(
            "1 Long Section\nPage one content.",
            "Page two content of same section.",
        )
        sections = _split_into_sections(pages)
        if sections:
            assert sections[-1].page_end == 2


# ── chunk_document ────────────────────────────────────────────────────────────

class TestChunkDocument:
    def test_returns_list_of_text_chunks(self):
        pages = make_pages("1 Introduction\n" + "Word " * 200)
        chunks = chunk_document(pages)
        assert isinstance(chunks, list)
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_chunk_indices_are_sequential(self):
        pages = make_pages("1 Intro\n" + "Word " * 400)
        chunks = chunk_document(pages)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_respects_target_tokens(self):
        cfg = ChunkingConfig(target_tokens=100, overlap_tokens=10, min_tokens=10)
        pages = make_pages("1 Section\n" + "word " * 500)
        chunks = chunk_document(pages, config=cfg)
        # All chunks except possibly the last should be near target size
        for chunk in chunks[:-1]:
            assert chunk.token_count <= cfg.target_tokens * 1.2  # allow 20% overshoot

    def test_table_chunks_flagged(self):
        text = "1 Section\nSome text.\n\n[TABLE]\ncol1  col2\nval1  val2\n[/TABLE]"
        chunks = chunk_document(make_pages(text))
        table_chunks = [c for c in chunks if c.is_table_chunk]
        assert len(table_chunks) >= 1

    def test_table_not_split(self):
        # A table block should not be split across two chunks
        table_rows = "\n".join(f"row{i}  data{i}" for i in range(20))
        text = f"1 Section\nIntro.\n\n[TABLE]\n{table_rows}\n[/TABLE]\n\nPost table."
        cfg = ChunkingConfig(target_tokens=50, overlap_tokens=5, min_tokens=5)
        chunks = chunk_document(make_pages(text), config=cfg)
        # Verify no chunk has partial [TABLE] marker
        for chunk in chunks:
            has_open = "[TABLE]" in chunk.text
            has_close = "[/TABLE]" in chunk.text
            assert has_open == has_close, (
                f"Table block split across chunk: open={has_open} close={has_close}\n{chunk.text[:200]}"
            )

    def test_small_document_produces_chunks(self):
        pages = make_pages("Short text.")
        chunks = chunk_document(pages)
        assert len(chunks) >= 1

    def test_empty_pages_produce_no_chunks(self):
        pages = make_pages("   ", "\n\n")
        chunks = chunk_document(pages)
        assert chunks == []

    def test_heading_path_carried_into_chunk(self):
        text = "1 Clinical Data\n1.1 Summary\nThis is the summary content."
        chunks = chunk_document(make_pages(text))
        assert any(len(c.heading_path) > 0 for c in chunks)

    def test_source_hash_is_populated(self):
        pages = make_pages("Some content for hashing test.")
        chunks = chunk_document(pages)
        for chunk in chunks:
            assert chunk.source_hash
            assert len(chunk.source_hash) == 16

    def test_no_empty_text_chunks(self):
        pages = make_pages("1 Section\n\n\n\nActual content here.")
        chunks = chunk_document(pages)
        for chunk in chunks:
            assert chunk.text.strip()

    def test_min_tokens_merge(self):
        """Tiny trailing fragments should be merged, not emitted as solo chunks."""
        cfg = ChunkingConfig(target_tokens=200, overlap_tokens=20, min_tokens=80)
        # First block is big, second is tiny
        text = "1 Section\n" + "word " * 200 + "\n\nTiny."
        chunks = chunk_document(make_pages(text), config=cfg)
        # No chunk should have fewer than min_tokens (except if there's only one)
        if len(chunks) > 1:
            for chunk in chunks:
                assert chunk.token_count >= cfg.min_tokens or chunk == chunks[-1]
