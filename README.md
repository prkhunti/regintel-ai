# RegIntel AI

Audit-ready clinical document intelligence for regulated AI workflows.

RegIntel AI is a portfolio / reference implementation of a document intelligence system for clinical and regulatory documents. It ingests documents, chunks and indexes source evidence, answers questions with citations, assigns confidence and risk signals, and records audit events for traceability.

> This repository is published as a portfolio and reference implementation. It is not currently accepting external contributions, and it should not be treated as a certified medical, regulatory, or production deployment.

## Problem

Clinical and regulatory teams need document question-answering systems that preserve provenance. A useful answer is not enough: reviewers need to know which source chunks support the answer, whether evidence was insufficient, how confident the system was, and what happened during ingestion, query, and evaluation flows.

RegIntel AI demonstrates one approach to that problem with a small, inspectable stack:

- document ingestion with PDF parsing, OCR fallback, and section-aware chunking
- hybrid retrieval using dense vectors, BM25, reciprocal rank fusion, and optional reranking
- structured answer generation with citation mapping and refusal support
- confidence scoring from retrieval, citation, and coverage signals
- audit events for important document, query, and evaluation actions
- offline evaluation utilities for retrieval and answer quality checks

## Key Features

- **Document ingestion**: asynchronous Celery ingestion for uploaded documents, PDF parsing through `pdfplumber`, optional OCR fallback, and table-aware chunking.
- **Evidence retrieval**: pgvector-backed dense retrieval, BM25 sparse retrieval, hybrid fusion, and pluggable reranker backends.
- **Grounded answers**: provider abstraction for OpenAI, Anthropic, and mock local mode, with Pydantic-validated structured output.
- **Confidence and risk**: scoring based on top retrieved evidence, citation density, retrieval score, and answer coverage.
- **Audit trail**: persisted events for uploads, processing, answer generation or refusal, and evaluation runs.
- **Evaluation harness**: pytest-backed retrieval quality checks and reusable metric code for recall, precision, MRR, citation recall, and refusal accuracy.
- **Web UI**: Next.js interface for querying, document workflows, audit inspection, and evaluation views.

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                         Next.js web app                         │
│        Query UI · Documents · Audit Log · Evaluation UI         │
└────────────────────────────┬────────────────────────────────────┘
                             │ REST
┌────────────────────────────▼────────────────────────────────────┐
│                        FastAPI service                          │
│   /documents · /query · /audit/events · /eval/runs · /health    │
└──────┬────────────────┬───────────────────────┬─────────────────┘
       │                │                       │
┌──────▼──────┐  ┌──────▼──────┐        ┌──────▼──────┐
│   Celery    │  │  Retrieval  │        │  LLM client │
│ ingestion   │  │  pipeline   │        │ OpenAI /    │
│   worker    │  │ Dense+BM25  │        │ Anthropic / │
└──────┬──────┘  │ Hybrid+rank │        │ mock        │
       │         └──────┬──────┘        └─────────────┘
       │                │
┌──────▼────────────────▼──────────────────────────────┐
│                 PostgreSQL + pgvector                 │
│ documents · chunks · embeddings · queries · answers   │
│ citations · eval runs · audit events                  │
└───────────────────────────────────────────────────────┘
```

## Tech Stack

- **API**: Python 3.12, FastAPI, SQLAlchemy async, Alembic, Pydantic
- **Worker**: Celery, Redis, pdfplumber, pytesseract, tiktoken
- **Retrieval**: pgvector, BM25 (`rank-bm25`), reciprocal rank fusion, optional rerankers
- **LLM providers**: OpenAI, Anthropic, mock mode for local development and CI
- **Web**: Next.js 15, React 19, TypeScript, Tailwind CSS
- **Infrastructure**: Docker Compose, PostgreSQL 16 with pgvector, Redis
- **Testing**: pytest, pytest-asyncio, ruff, mypy

## Repository Structure

```text
regintel-ai/
├── apps/
│   ├── api/                 # FastAPI app, routers, services, models, migrations
│   ├── web/                 # Next.js UI
│   └── worker/              # Celery ingestion worker
├── packages/
│   ├── retrieval/           # parsing, chunking, dense/sparse retrieval, fusion, reranking
│   ├── evals/               # evaluation runner and metrics
│   ├── schemas/             # shared Pydantic schemas
│   └── prompts/templates/   # answer prompts
├── data/
│   ├── gold_queries/        # small offline eval fixture set
│   └── sample_docs/         # sample local documents
├── docs/
│   ├── architecture/        # architecture notes and diagrams (placeholder)
│   ├── decisions/           # ADRs / design notes (placeholder)
│   ├── eval-results/        # generated eval outputs, ignored by default
│   └── threat-model/        # threat model notes (placeholder)
├── infra/
│   ├── docker/              # Dockerfiles
│   ├── docker-compose.yaml  # base Compose stack
│   └── docker-compose.dev.yaml
├── scripts/                 # local development helpers
├── tests/                   # unit, integration, and eval tests
├── Makefile                 # common local commands
└── pyproject.toml           # Python dependency groups and tool config
```

## Local Setup

### Prerequisites

- Docker Desktop with Docker Compose v2
- `make`
- Optional for non-Docker Python workflows: Python 3.12
- Optional for non-Docker web workflows: Node.js 20+

### First Run

```bash
git clone https://github.com/your-username/regintel-ai.git
cd regintel-ai
make env
make up-build
make migrate
```

Then open:

- Web app: <http://localhost:3000>
- API docs: <http://localhost:8000/docs>
- Health check: <http://localhost:8000/api/v1/health>

The default `.env.example` uses `LLM_PROVIDER=random`, which routes LLM calls to a mock client and embeddings to deterministic random vectors. This keeps the stack runnable without API keys for inspection and local development. Use real provider keys only when you intend to call external APIs.

### Development Mode

```bash
make dev
make migrate
```

Development mode bind-mounts the source tree and enables hot reload for FastAPI and Next.js. Worker code changes require a worker restart:

```bash
make dev-restart-worker
```

## Environment Variables

Copy `.env.example` to `.env` with `make env`. The main variables are:

| Variable | Purpose |
| --- | --- |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | PostgreSQL container credentials for local Compose |
| `DATABASE_URL` | SQLAlchemy async database URL |
| `REDIS_URL` | Redis URL for health checks and shared runtime access |
| `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND` | Celery broker and result backend URLs |
| `LLM_PROVIDER` | `random`, `mock`, `openai`, or `anthropic` |
| `LLM_MODEL` | Chat model name for the selected remote provider |
| `OPENAI_API_KEY` | Required for OpenAI chat or OpenAI embeddings |
| `ANTHROPIC_API_KEY` | Required when `LLM_PROVIDER=anthropic` |
| `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `EMBEDDING_BATCH_SIZE` | Embedding backend configuration |
| `RERANKER_BACKEND` | `none`, `cross-encoder`, or `cohere` |
| `COHERE_API_KEY` | Required only when using the Cohere reranker |
| `UPLOAD_DIR`, `INDEX_DIR` | Runtime storage paths for uploads and BM25 indexes |
| `CHUNK_TARGET_TOKENS`, `CHUNK_OVERLAP_TOKENS`, `OCR_TEXT_THRESHOLD` | Ingestion and chunking controls |
| `APP_ENV`, `SECRET_KEY`, `LOG_LEVEL` | Application runtime settings |
| `API_URL` | Server-side Next.js rewrite target for the API |

Do not commit `.env`, real API keys, credentials, private endpoints, or private datasets.

## Common Commands

```bash
make help                 # list available commands
make up-build             # build and start the base stack
make dev                  # start the hot-reload development stack
make migrate              # run Alembic migrations
make logs                 # follow all service logs
make test                 # run unit and eval tests in the API container
make lint                 # run ruff in the API container
make typecheck            # run mypy in the API container
make down                 # stop services
make down-v               # stop services and remove volumes
```

The Makefile commands expect the Docker Compose stack to be running for test, lint, typecheck, and migration commands.

## Testing

```bash
make test-unit
make test-eval
make test
```

The unit and eval tests are intended to run offline inside the API container. The eval suite uses the small fixture set in `data/gold_queries/sample_queries.json`; it is useful as a regression check, not as a benchmark claim.

## Screenshots And Demo

Screenshots and demo media are not included yet.

TODO:

- add a query workflow screenshot
- add a document upload / processing screenshot
- add an audit log screenshot
- add a short demo GIF after the UI is stable

## Architecture Notes

`docs/architecture/`, `docs/decisions/`, and `docs/threat-model/` are reserved for deeper design documentation. They currently contain placeholders only.

TODO:

- add an exported architecture diagram image
- document the threat model for prompt injection, private data handling, and audit integrity
- add concise ADRs for retrieval, LLM provider abstraction, and confidence scoring

## Current Status And Limitations

- This is an early reference implementation, not a certified clinical, regulatory, or medical device system.
- The default local mode uses mock LLM responses and deterministic random embeddings.
- External provider use requires valid API keys and may incur provider costs.
- Role-based access control, production authentication, deployment hardening, and private dataset governance are not implemented.
- Sample data and eval fixtures are intentionally small and should not be interpreted as benchmark evidence.
- Several documentation folders are placeholders until diagrams, decisions, and threat-model notes are added.

## Contributing

This repository is published primarily as a portfolio and reference implementation. External issues and pull requests are not currently being accepted.

You may read, clone, and use the project subject to the license.

## Security

Do not publish secrets, credentials, private endpoints, protected health information, or private datasets in this repository. If you find a security issue, contact the maintainer directly rather than opening a public issue.

## Maintainer

Maintained by Prakash Khunti.

For portfolio or professional inquiries, use the contact channel listed on the maintainer's GitHub profile.
