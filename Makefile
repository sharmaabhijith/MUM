.PHONY: install test lint generate-all generate-scenario validate validate-docs eval eval-all eval-model eval-report eval-compare eval-export list-models cost-estimate dry-run export clean

# ── Setup ──────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check datagen/ memory/ tests/

# ── Data Generation ────────────────────────────────────────────────
generate-all:
	python -m datagen.cli generate --all

generate-scenario:
	python -m datagen.cli generate --scenario $(SCENARIO)

validate:
	python -m datagen.cli validate --scenario $(SCENARIO)

validate-docs:
	python -m datagen.cli validate-docs --scenario $(SCENARIO)

cost-estimate:
	python -m datagen.cli estimate-cost --all

dry-run:
	python -m datagen.cli generate --scenario 1 --dry-run

export:
	python -m datagen.cli export --format huggingface

# ── Memory Evaluation ──────────────────────────────────────────────

# Full grid: all 7 models x all 5 methods x all 5 scenarios
eval-all:
	python -m memory.cli evaluate --all-scenarios --method all --answer-model all

# Single scenario, all models/methods
eval:
	python -m memory.cli evaluate --scenario $(SCENARIO) --method all --answer-model all

# Single model, all methods/scenarios
eval-model:
	python -m memory.cli evaluate --all-scenarios --method all --answer-model $(MODEL)

# Reports
eval-report:
	python -m memory.cli report --scenario $(SCENARIO)

eval-compare:
	python -m memory.cli compare

eval-export:
	python -m memory.cli export-results

list-models:
	python -m memory.cli list-models

# ── Cleanup ────────────────────────────────────────────────────────
clean:
	rm -rf MUMBench/results/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
