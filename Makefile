SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: help lint test eval eval-fast verify-gates ragas api frontend build clean

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

lint: ## Ruff, same scope as CI
	ruff check src tests eval alembic

test: ## Fast tests with the local coverage gate
	pytest tests/ -v -m "not slow" --cov=src --cov-report=term-missing --cov-fail-under=70

# ---------------------------------------------------------------------------
# Quality gates. Local-only by decision (issue #110): both need paid API keys,
# and Voyage's 3 RPM trial cap means the retrieval gate takes ~13 minutes of
# pacing. Do not wire these into CI — CI runs `lint` and `test` only.
# ---------------------------------------------------------------------------

eval: verify-gates ## Every gate that runs on this account (alias for verify-gates)
	@echo "All local gates passed."

verify-gates: ## Run both enforced gates: citation gates + retrieval precision
	@echo "== citation gates (eval/verify_eval.py) =="
	@python3 eval/verify_eval.py && \
	 echo "" && \
	 echo "== retrieval precision (eval/retrieval_precision.py) ==" && \
	 python3 eval/retrieval_precision.py

eval-fast: ## Citation gates only — skips the ~13 min retrieval run
	@python3 eval/verify_eval.py

ragas: ## Ragas metrics. Requires funded Groq + Voyage keys; cannot complete on free tiers.
	python3 eval/eval_runner.py

api: ## Run the FastAPI server on :8001
	uvicorn src.api.app:app --host 0.0.0.0 --port 8001

frontend: ## Run the Vite dev server on :5173
	cd frontend && npm run dev

build: ## Production frontend build
	cd frontend && npm run build

clean: ## Remove caches and coverage output
	rm -rf .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +