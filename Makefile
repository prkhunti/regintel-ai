.PHONY: help up up-build down down-v build logs logs-api logs-worker \
        migrate migrate-new db-shell redis-cli \
        test test-unit test-integration test-eval \
        lint format typecheck eval clean

COMPOSE := docker compose -f infra/docker/docker-compose.yml
PYTHON  := python3

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' | sort

# ── Docker ───────────────────────────────────────────────────────────────────
up: ## Start all services (detached)
	$(COMPOSE) up -d

up-build: ## Rebuild images and start all services
	$(COMPOSE) up -d --build

down: ## Stop all services
	$(COMPOSE) down

down-v: ## Stop all services and remove volumes (destructive)
	$(COMPOSE) down -v

build: ## Build all Docker images
	$(COMPOSE) build

logs: ## Tail logs for all services
	$(COMPOSE) logs -f

logs-api: ## Tail API logs
	$(COMPOSE) logs -f api

logs-worker: ## Tail worker logs
	$(COMPOSE) logs -f worker

# ── Database ─────────────────────────────────────────────────────────────────
migrate: ## Run database migrations
	$(COMPOSE) exec api alembic upgrade head

migrate-new: ## Create a new migration (usage: make migrate-new MSG="add table")
	$(COMPOSE) exec api alembic revision --autogenerate -m "$(MSG)"

db-shell: ## Open a psql shell
	$(COMPOSE) exec db psql -U regintel -d regintel

redis-cli: ## Open a Redis CLI
	$(COMPOSE) exec redis redis-cli

# ── Testing ───────────────────────────────────────────────────────────────────
test: ## Run all tests inside Docker
	$(COMPOSE) exec api python -m pytest tests/ -v

test-unit: ## Run unit tests
	$(COMPOSE) exec api python -m pytest tests/unit/ -v

test-integration: ## Run integration tests
	$(COMPOSE) exec api python -m pytest tests/integration/ -v

test-eval: ## Run the evaluation harness
	$(COMPOSE) exec api python -m pytest tests/eval/ -v --tb=short

# ── Code quality ─────────────────────────────────────────────────────────────
lint: ## Lint Python code with ruff
	$(COMPOSE) exec api ruff check /app /app/../packages

format: ## Format Python code with ruff
	$(COMPOSE) exec api ruff format /app /app/../packages

typecheck: ## Type-check Python code with mypy
	$(COMPOSE) exec api mypy /app/app /app/../packages

# ── Eval ─────────────────────────────────────────────────────────────────────
eval: ## Run full evaluation suite and write results
	$(COMPOSE) exec api python -m packages.evals.runner --output docs/eval-results/latest.json

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean: ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
