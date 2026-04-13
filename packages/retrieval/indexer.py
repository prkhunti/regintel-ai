"""
Keyword (sparse) index for hybrid retrieval.

Two implementations behind a common interface:

BM25Index
    Pure Python, backed by rank_bm25.  Best for testing and small corpora
    (< ~50k chunks).  Serialisable to disk via pickle.

PostgresFTSIndexer
    Production path.  Uses PostgreSQL tsvector / ts_rank_cd for keyword
    scoring.  No extra service required — runs in the same DB as pgvector.
    Integrates with the existing Chunk ORM model via a tsvector column
    populated by a trigger or updated at index time.
"""
from __future__ import annotations

import logging
import pickle
import re
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

logger = logging.getLogger(__name__)

# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SparseHit:
    chunk_id: str           # UUID string or integer index
    score: float
    text: str | None = None  # populated when available


# ── Shared text preprocessing ─────────────────────────────────────────────────

_PUNCT = re.compile(r"[" + re.escape(string.punctuation) + r"]")
_WHITESPACE = re.compile(r"\s+")

_STOPWORDS = frozenset(
    "a an the and or but in on at to for of with by from is are was were be "
    "been being have has had do does did will would could should may might "
    "shall this that these those it its".split()
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    tokens = _WHITESPACE.split(text.strip())
    return [t for t in tokens if t and t not in _STOPWORDS]


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseIndex(ABC):
    @abstractmethod
    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        """Build the index from a list of (id, text) pairs."""

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[SparseHit]:
        """Return up to top_k hits for the query, sorted by descending score."""

    @abstractmethod
    def add(self, chunk_id: str, text: str) -> None:
        """Incrementally add a single document to the index."""


# ── BM25 (in-memory) ──────────────────────────────────────────────────────────

class BM25Index(BaseIndex):
    """
    In-memory BM25 index backed by rank_bm25.

    Usage
    -----
    idx = BM25Index()
    idx.build(chunk_ids, texts)
    hits = idx.search("cybersecurity risk control", top_k=10)
    idx.save("/tmp/bm25.pkl")
    idx = BM25Index.load("/tmp/bm25.pkl")
    """

    def __init__(self) -> None:
        self._chunk_ids: list[str] = []
        self._texts: list[str] = []
        self._bm25 = None  # BM25Okapi instance

    # ── Public interface ──────────────────────────────────────────────────────

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must have the same length")
        self._chunk_ids = list(chunk_ids)
        self._texts = list(texts)
        self._bm25 = self._make_bm25(texts)
        logger.info("BM25 index built with %d documents", len(texts))

    def add(self, chunk_id: str, text: str) -> None:
        """
        Append a single document.  Re-builds the underlying BM25Okapi object —
        acceptable for incremental ingestion; not for high-frequency streaming.
        """
        self._chunk_ids.append(chunk_id)
        self._texts.append(text)
        self._bm25 = self._make_bm25(self._texts)

    def search(self, query: str, top_k: int = 10) -> list[SparseHit]:
        if self._bm25 is None or not self._chunk_ids:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Pair up and sort descending
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for i, score in indexed[:top_k]:
            if score <= 0:
                break
            results.append(SparseHit(
                chunk_id=self._chunk_ids[i],
                score=float(score),
                text=self._texts[i],
            ))
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {"chunk_ids": self._chunk_ids, "texts": self._texts},
                f,
            )
        logger.info("BM25 index saved to %s (%d docs)", path, len(self._chunk_ids))

    @classmethod
    def load(cls, path: Path | str) -> "BM25Index":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found: {path}")
        with path.open("rb") as f:
            data = pickle.load(f)
        idx = cls()
        idx.build(data["chunk_ids"], data["texts"])
        logger.info("BM25 index loaded from %s (%d docs)", path, len(idx._chunk_ids))
        return idx

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _make_bm25(texts: list[str]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise RuntimeError("rank_bm25 is required: pip install rank-bm25") from e
        tokenized = [_tokenize(t) for t in texts]
        return BM25Okapi(tokenized)


# ── PostgreSQL FTS indexer ────────────────────────────────────────────────────

class PostgresFTSIndexer(BaseIndex):
    """
    Production keyword indexer using PostgreSQL full-text search.

    The ``chunks`` table is extended with a GIN-indexed ``fts`` column
    (type tsvector).  This indexer populates that column at ingestion time
    and queries it at retrieval time using ``ts_rank_cd``.

    ``build()`` and ``add()`` operate on an open SQLAlchemy Session.
    ``search()`` issues a raw SQL query and returns SparseHit objects.
    """

    def __init__(self, session=None) -> None:
        """
        Args:
            session: SQLAlchemy Session (sync) or AsyncSession.
                     Can be set later via ``set_session()``.
        """
        self._session = session

    def set_session(self, session) -> None:
        self._session = session

    # ── DDL helper ────────────────────────────────────────────────────────────

    @staticmethod
    def ensure_fts_column(session) -> None:
        """
        Idempotently add the ``fts`` tsvector column + GIN index to ``chunks``.
        Call this once during migrations or app startup.
        """
        session.execute("""
            ALTER TABLE chunks
            ADD COLUMN IF NOT EXISTS fts tsvector
                GENERATED ALWAYS AS (
                    to_tsvector('english', coalesce(text, ''))
                ) STORED;
        """)
        session.execute("""
            CREATE INDEX IF NOT EXISTS ix_chunks_fts
            ON chunks USING GIN (fts);
        """)
        session.commit()
        logger.info("FTS column and index ensured on chunks table")

    # ── BaseIndex interface ───────────────────────────────────────────────────

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        """
        Bulk-update the ``fts`` column for the given chunk IDs.
        The GENERATED ALWAYS column handles this automatically on INSERT/UPDATE,
        so this is a no-op when the generated column DDL is in place.
        """
        logger.info(
            "PostgresFTSIndexer.build(): fts column is auto-generated; "
            "%d chunks will be indexed automatically on next VACUUM/analyze.",
            len(chunk_ids),
        )

    def add(self, chunk_id: str, text: str) -> None:
        """No-op — the generated column updates on row INSERT."""

    def search(self, query: str, top_k: int = 10) -> list[SparseHit]:
        if self._session is None:
            raise RuntimeError("PostgresFTSIndexer requires a database session. Call set_session() first.")

        tsquery = _to_tsquery(query)
        if not tsquery:
            return []

        sql = """
            SELECT id::text, text,
                   ts_rank_cd(fts, plainto_tsquery('english', :q)) AS score
            FROM   chunks
            WHERE  fts @@ plainto_tsquery('english', :q)
            ORDER  BY score DESC
            LIMIT  :k
        """
        rows = self._session.execute(sql, {"q": query, "k": top_k}).fetchall()
        return [
            SparseHit(chunk_id=row.id, score=float(row.score), text=row.text)
            for row in rows
        ]


def _to_tsquery(text: str) -> str:
    """Convert free text to a simple tsquery-safe string."""
    tokens = _tokenize(text)
    return " & ".join(tokens) if tokens else ""


# ── Factory ───────────────────────────────────────────────────────────────────

def get_indexer(backend: str = "bm25", **kwargs) -> BaseIndex:
    """
    Factory for keyword indexers.

    Args:
        backend: ``"bm25"`` (default, in-memory) or ``"postgres"`` (production).
        **kwargs: Passed to the indexer constructor.
    """
    if backend == "bm25":
        return BM25Index()
    if backend == "postgres":
        return PostgresFTSIndexer(**kwargs)
    raise ValueError(f"Unknown indexer backend: {backend!r}. Choose 'bm25' or 'postgres'.")
