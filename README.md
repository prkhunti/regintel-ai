# RegIntel AI

**Audit-ready clinical document intelligence for regulated medical device environments.**

RegIntel ingests clinical and regulatory documents (CERs, risk management files, IFUs, software requirements), retrieves grounded evidence with hybrid search, generates traceable answers with inline citations, scores confidence, and maintains a full immutable audit trail. Every design decision reflects the constraints of high-trust, regulated AI — where hallucinations, traceability failures, and undetected refusals are unacceptable.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Client (Next.js)                       │
│         Query UI  ·  Audit Log  ·  Evaluation Dashboard         │
└────────────────────────────┬────────────────────────────────────┘
                             │ REST
┌────────────────────────────▼────────────────────────────────────┐
│                      FastAPI  (apps/api)                        │
│  /documents  ·  /query  ·  /audit/events  ·  /eval/runs        │
└──────┬────────────────┬───────────────────────┬─────────────────┘
       │                │                       │
┌──────▼──────┐  ┌──────▼──────┐        ┌──────▼──────┐
│  Ingestion  │  │  Retrieval  │        │  LLM Client │
│   Worker    │  │   Engine    │        │  (OpenAI /  │
│  (Celery)   │  │             │        │  Anthropic) │
└──────┬──────┘  │  Dense      │        └─────────────┘
       │         │  Sparse     │
       │         │  Hybrid RRF │
       │         │  Reranker   │
       │         └──────┬──────┘
       │                │
┌──────▼────────────────▼──────────────────────────────┐
│              PostgreSQL + pgvector  (+ Redis)        │
│  documents · chunks · embeddings (HNSW) · queries   │
│  responses · citations · eval_runs · audit_events   │
└──────────────────────────────────────────────────────┘
```

---

## Features

### Document ingestion
- PDF parsing with `pdfplumber`; OCR fallback via `pytesseract` when extracted text falls below a configurable character threshold
- Table-block protection — content between `[TABLE]…[/TABLE]` markers is kept as an atomic unit and never split mid-row
- Section-aware chunking: sentence-boundary splits, heading inheritance, configurable token targets (`ChunkingConfig`)
- Per-chunk provenance: `section_title`, `heading_path`, `page_start/end`, `token_count`, `source_hash`
- Celery worker handles ingestion asynchronously; HTTP response is immediate (`202 Accepted`)

### Hybrid retrieval
| Component | Implementation |
|-----------|----------------|
| Dense retrieval | pgvector `vector_cosine_ops` with HNSW index (`m=16, ef_construction=64`) |
| Sparse retrieval | BM25 (`rank-bm25`) — per-document `.bm25.pkl` files, merged at query time |
| Fusion | Reciprocal Rank Fusion (RRF) or weighted alpha blend |
| Reranking | `CrossEncoderReranker` (sentence-transformers), `CohereReranker`, or `IdentityReranker` |

Dense and sparse retrieval run concurrently with `asyncio.gather`. BM25 scores are normalised to [0, 1] before fusion.

### Grounded answer generation
- Provider-abstracted LLM client: **OpenAI** (JSON mode) and **Anthropic** (tool use) behind a common `BaseLLMClient` interface
- Structured output via Pydantic — `StructuredAnswerOutput` validated before use; no regex parsing
- Prompt (`answer_structured.txt`) enforces: cite every claim, refuse when context is insufficient, return empty answer on refusal
- `AnswerGenerator` maps `CitationOutput.chunk_index` back to the source `DenseHit` for full provenance

### Confidence scoring
Four signals combined into a single `[0, 1]` score:

| Signal | Weight | Description |
|--------|--------|-------------|
| `top_chunk_score` | 0.30 | Max cosine similarity of the best retrieved chunk |
| `citation_density` | 0.35 | Fraction of answer sentences containing at least one `[N]` marker |
| `retrieval_score` | 0.25 | Mean score across the top-k chunks |
| `coverage_ratio` | 0.10 | Fraction of answer content terms that appear in source chunks |

Risk levels: `LOW` (≥ 0.75) · `MEDIUM` (≥ 0.50) · `HIGH` (≥ 0.30) · `CRITICAL` (< 0.30). Refusals short-circuit to `confidence=0.0, risk=CRITICAL`.

### Audit trail
- `audit_events` table records every significant system action: `document_uploaded`, `document_processed`, `answer_generated`, `answer_refused`, `eval_run_started`, `eval_run_completed`, `pii_detected`, `injection_detected`
- Every event carries `event_type`, `resource_type`, `resource_id`, `actor`, `detail` (JSON), and `created_at`
- `GET /api/v1/audit/events` supports filtering by event type, resource type, resource ID, and date range

### Evaluation harness
- `EvalCase` records store a query, expected chunk IDs, and an `is_insufficient` flag
- `EvalRunner` runs the full pipeline (retrieve → generate → score) per case and returns aggregate metrics:
  - **Retrieval**: Recall@10, Precision@10, MRR
  - **Answer quality**: citation recall (fraction of expected chunks cited), refusal accuracy
- `POST /api/v1/eval/runs` triggers a synchronous run and persists results to `eval_runs`
- Offline pytest harness in `tests/eval/` covers BM25 quality, hybrid fusion quality, and metrics correctness against the gold query set

---

## Repository structure

```
regintel-ai/
├── apps/
│   ├── api/                    # FastAPI application
│   │   ├── app/
│   │   │   ├── models/         # SQLAlchemy ORM models
│   │   │   ├── routers/        # documents · query · audit · eval
│   │   │   └── services/       # answer_service · confidence · llm_client · document_service
│   │   └── migrations/         # Alembic (001 schema, 002 HNSW index)
│   ├── web/                    # Next.js 15 frontend (TypeScript + Tailwind)
│   │   └── src/
│   │       ├── pages/          # index (query) · audit · eval
│   │       ├── components/     # AnswerCard · CitationList · ConfidenceBadge
│   │       └── lib/api.ts      # typed fetch client
│   └── worker/                 # Celery ingestion worker
│       └── tasks/ingestion.py
├── packages/
│   ├── retrieval/              # chunker · parser · dense · sparse · hybrid · reranker · embedder
│   ├── evals/                  # metrics · runner
│   ├── schemas/                # Pydantic schemas (document · query · response · eval · audit · llm_output)
│   └── prompts/templates/      # answer_structured.txt · answer_grounded.txt
├── data/
│   └── gold_queries/
│       └── sample_queries.json # 12 corpus chunks + 15 labelled queries for offline eval
├── tests/
│   ├── unit/                   # test_chunker · test_retrieval · test_confidence
│   └── eval/                   # test_retrieval_quality (BM25 + hybrid + metrics)
└── infra/
    └── docker/                 # docker-compose.yml · Dockerfile.api · Dockerfile.worker
```

---

## API reference

### Documents
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/documents/upload` | Upload a PDF; enqueues ingestion |
| `POST` | `/api/v1/documents/{id}/process` | Re-trigger ingestion |
| `GET`  | `/api/v1/documents` | List all documents |
| `GET`  | `/api/v1/documents/{id}` | Document detail + chunk count |
| `GET`  | `/api/v1/documents/{id}/chunks` | Paginated chunk list |

### Query
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/query` | Full pipeline: retrieve → rerank → generate → score → persist |
| `GET`  | `/api/v1/query/{id}` | Fetch stored query by ID |

### Audit
| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/v1/audit/events` | Paginated event list (filterable) |
| `GET`  | `/api/v1/audit/events/count` | Count matching events |
| `GET`  | `/api/v1/audit/events/{id}` | Single event |

### Evaluation
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/eval/cases` | Create an eval case |
| `GET`  | `/api/v1/eval/cases` | List eval cases |
| `DELETE` | `/api/v1/eval/cases/{id}` | Delete a case |
| `POST` | `/api/v1/eval/runs` | Trigger a run over all (or selected) cases |
| `GET`  | `/api/v1/eval/runs` | List completed runs |
| `GET`  | `/api/v1/eval/runs/{id}` | Run detail |

---

## Local setup

### Prerequisites

- Docker Desktop (includes Docker Compose v2)

### First-time setup

```bash
git clone https://github.com/your-username/regintel-ai.git
cd regintel-ai
make env          # copies .env.example → .env
```

Open `.env` and set your API keys when you have them.
`LLM_PROVIDER=random` is the default — the full ingestion pipeline runs
with deterministic random embeddings and a mock LLM so you can develop and
test without any API credits.

### Production mode (baked images, no bind mounts)

```bash
make up-build     # build images and start all 5 services
make migrate      # run database migrations (first time only)
# Web UI  → http://localhost:3000
# API docs → http://localhost:8000/docs
```

### Dev mode (bind-mounted source + hot reload)

```bash
make dev          # build dev images and start the dev stack
make migrate      # run migrations if not already done
# Web UI  → http://localhost:3000  (Next.js HMR active)
# API docs → http://localhost:8000/docs  (uvicorn --reload active)
```

- `apps/api/**/*.py` changes reload uvicorn automatically.
- `apps/web/**/*.tsx` changes reload Next.js automatically.
- `apps/worker/**/*.py` changes require `make dev-restart-worker`.

```bash
make dev-logs           # tail all dev logs
make dev-restart-worker # apply worker code changes
make dev-down           # stop dev stack
```

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | `random` (stub, no key needed) · `openai` · `anthropic` | `random` |
| `LLM_MODEL` | Model name | `gpt-4o` |
| `OPENAI_API_KEY` | OpenAI key (required when `LLM_PROVIDER=openai`) | — |
| `ANTHROPIC_API_KEY` | Anthropic key (required when `LLM_PROVIDER=anthropic`) | — |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `DATABASE_URL` | Async PostgreSQL URL | `postgresql+asyncpg://...` |
| `REDIS_URL` | Redis connection string | `redis://redis:6379/0` |
| `RERANKER_BACKEND` | `none` · `cross-encoder` · `cohere` | `none` |

---

## Running tests

```bash
make test-unit    # chunker, retrieval metrics, confidence scorer
make test-eval    # BM25 quality, hybrid fusion quality, metrics correctness
make test         # all tests
```

The eval suite runs entirely offline — no database or API key required. Dense retrieval is replaced with a perfect oracle and BM25 is built in-memory from the gold corpus.

### Benchmark targets (gold query set, 15 queries)

| Metric | Target |
|--------|--------|
| BM25 Recall@10 | ≥ 0.50 |
| Hybrid Recall@10 (oracle dense) | ≥ 0.80 |
| BM25 MRR | ≥ 0.40 |
| Hybrid MRR | ≥ BM25 MRR |

---

## Design decisions

**Hybrid retrieval over dense-only.** BM25 handles exact-match queries (standard numbers, product codes) that embedding models often miss. RRF fusion with `k=60` is robust to score-scale differences and outperforms both retrievers individually on the benchmark.

**pgvector over a dedicated vector database.** Clinical document corpora are small (thousands to low millions of chunks). A single Postgres instance with an HNSW index handles the load without operational overhead. HNSW index creation uses `CREATE INDEX CONCURRENTLY` in a separate migration so it does not block the schema migration transaction.

**Section-aware chunking.** Naive fixed-size chunking splits sentences mid-way and loses section context. The chunker respects sentence boundaries, protects table blocks atomically, and inherits section headings into each chunk's metadata. This significantly improves retrieval precision for structured regulatory documents.

**Structured LLM output via Pydantic.** The answer prompt returns a validated `StructuredAnswerOutput` object — not free text with regex-parsed citations. OpenAI uses `response_format={"type": "json_object"}`; Anthropic uses tool use with `model_json_schema()` as the input schema. No citation parsing can fail silently.

**Provider abstraction.** `BaseLLMClient` defines `complete()` and `complete_structured()`. Swapping providers is a one-line config change. This also makes the LLM layer fully mockable in tests.

**Answer refusal on insufficient evidence.** When the retrieved context does not support an answer, the model sets `insufficient_context=true` and returns an empty answer string. This is enforced in the prompt and validated by Pydantic — not left to model discretion.

---

## Roadmap

**V2**
- Contradiction detection between document versions
- Traceability mapper: requirement → evidence → source chunk
- Gap analysis: given a checklist, identify missing supporting evidence
- Adversarial eval set (prompt injection, indirect injection via documents)

**V3**
- Streaming answers via Server-Sent Events
- Human feedback loop with active learning for retrieval fine-tuning
- Role-based access control (reviewer, uploader, auditor)
- Local model serving via vLLM for air-gapped environments
