.PHONY: help \
        env \
        up up-build down down-v build logs logs-api logs-worker \
        dev dev-build dev-down dev-logs dev-logs-api dev-logs-worker dev-restart-worker \
        migrate migrate-new db-shell redis-cli \
        test test-unit test-integration test-eval \
        lint format typecheck eval clean

COMPOSE     := docker compose -f infra/docker/docker-compose.yml
COMPOSE_DEV := docker compose -f infra/docker/docker-compose.yml \
                              -f infra/docker/docker-compose.dev.yml

# ── Help ─────────────────────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-26s\033[0m %s\n", $$1, $$2}' | sort

# ── First-time setup ──────────────────────────────────────────────────────────
env: ## Copy .env.example → .env (run once, then fill in API keys)
	@if [ -f .env ]; then \
	  echo ".env already exists — skipping. Delete it first to reset."; \
	else \
	  cp .env.example .env && echo "Created .env from .env.example. Edit it before running make up-build."; \
	fi

# ── Production (baked images, no bind mounts) ─────────────────────────────────
up: ## Start all services detached (uses pre-built images)
	$(COMPOSE) up -d

up-build: ## Rebuild images then start all services
	$(COMPOSE) up -d --build

down: ## Stop all services
	$(COMPOSE) down

down-v: ## Stop all services and remove volumes (destructive — wipes DB data)
	$(COMPOSE) down -v

build: ## Build all Docker images
	$(COMPOSE) build

logs: ## Tail logs for all services
	$(COMPOSE) logs -f

logs-api: ## Tail API logs
	$(COMPOSE) logs -f api

logs-worker: ## Tail worker logs
	$(COMPOSE) logs -f worker

# ── Dev mode (bind-mounted source + hot reload) ───────────────────────────────
dev: ## Start dev stack (bind-mounted source, uvicorn --reload, next dev)
	$(COMPOSE_DEV) up -d --build

dev-build: ## Rebuild dev images then start dev stack
	$(COMPOSE_DEV) build

dev-down: ## Stop dev stack
	$(COMPOSE_DEV) down

dev-logs: ## Tail logs for all dev services
	$(COMPOSE_DEV) logs -f

dev-logs-api: ## Tail dev API logs
	$(COMPOSE_DEV) logs -f api

dev-logs-worker: ## Tail dev worker logs
	$(COMPOSE_DEV) logs -f worker

dev-restart-worker: ## Restart worker after code changes (no hot reload in worker)
	$(COMPOSE_DEV) restart worker

# ── Database ─────────────────────────────────────────────────────────────────
migrate: ## Run database migrations (works against prod or dev stack)
	$(COMPOSE) exec api alembic upgrade head

migrate-new: ## Create a new migration (usage: make migrate-new MSG="add table")
	$(COMPOSE) exec api alembic revision --autogenerate -m "$(MSG)"

db-shell: ## Open a psql shell
	$(COMPOSE) exec db psql -U regintel -d regintel

redis-cli: ## Open a Redis CLI
	$(COMPOSE) exec redis redis-cli

# ── Testing ───────────────────────────────────────────────────────────────────
test: ## Run all tests inside the API container
	$(COMPOSE) exec api python -m pytest tests/ -v -p no:cacheprovider

test-unit: ## Run unit tests (chunker, retrieval, confidence — no DB or API key)
	$(COMPOSE) exec api python -m pytest tests/unit/ -v -p no:cacheprovider

test-integration: ## Run integration tests
	$(COMPOSE) exec api python -m pytest tests/integration/ -v -p no:cacheprovider

test-eval: ## Run evaluation harness (BM25 quality, hybrid fusion, metrics)
	$(COMPOSE) exec api python -m pytest tests/eval/ -v --tb=short -p no:cacheprovider

# ── Code quality ─────────────────────────────────────────────────────────────
lint: ## Lint Python code with ruff
	$(COMPOSE) exec api ruff check /app /app/packages

format: ## Format Python code with ruff
	$(COMPOSE) exec api ruff format /app /app/packages

typecheck: ## Type-check Python code with mypy
	$(COMPOSE) exec api mypy /app/app /app/packages

# ── Eval ─────────────────────────────────────────────────────────────────────
eval: ## Run full evaluation suite and write results
	$(COMPOSE) exec api python -m packages.evals.runner --output docs/eval-results/latest.json

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean: ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
