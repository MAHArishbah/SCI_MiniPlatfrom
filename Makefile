.PHONY: install dq pipeline train run test test-integration lint format up down assistant

install:            ## local dev environment
	pip install -r requirements-dev.txt

dq:                 ## standalone data quality check on the raw CSVs
	python dq_check.py --input data/raw --output artifacts/dq_report.json

pipeline:           ## build the DuckDB warehouse (idempotent)
	python -m sci.pipeline.build --input data/raw --db warehouse/sci.duckdb

train: pipeline     ## retrain the model and rewrite artifacts/model.joblib + model_card.json
	python -m sci.ml.train --db warehouse/sci.duckdb --out artifacts

run: dq pipeline    ## run the API locally on :8000
	uvicorn sci.api.main:app --reload --port 8000

test:               ## unit tests (no infrastructure)
	pytest tests/unit -q

test-integration:   ## needs the stack running (make up)
	SCI_BASE_URL=http://localhost:8000 pytest tests/integration -q -m integration

lint:
	ruff check . && ruff format --check .

format:
	ruff check . --fix && ruff format .

up:                 ## full stack in Docker
	docker compose up -d --build --wait

down:
	docker compose down -v

assistant:          ## GenAI assistant CLI (needs ANTHROPIC_API_KEY in .env)
	docker compose run --rm assistant
