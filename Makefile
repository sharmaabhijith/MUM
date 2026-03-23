.PHONY: install test lint generate-all generate-scenario validate eval cost-estimate dry-run export clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src/ eval/ tests/

generate-all:
	python -m src.cli generate --all

generate-scenario:
	python -m src.cli generate --scenario $(SCENARIO)

validate:
	python -m src.cli validate --scenario $(SCENARIO)

validate-docs:
	python -m src.cli validate-docs --scenario $(SCENARIO)

eval:
	python -m eval.run_eval --scenario $(SCENARIO) --baseline $(BASELINE)

cost-estimate:
	python -m src.cli estimate-cost --all

dry-run:
	python -m src.cli generate --scenario 1 --dry-run

export:
	python -m src.cli export --format huggingface

clean:
	rm -rf output/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
