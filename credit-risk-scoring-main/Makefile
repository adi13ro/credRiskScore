.PHONY: install pipeline dashboard test lint clean

install:
	uv sync --extra dev

pipeline:
	uv run creditrisk-pipeline

dashboard:
	uv run streamlit run app/dashboard.py

test:
	uv run pytest -q

lint:
	uv run ruff check src tests app

clean:
	rm -rf data/raw/* data/processed/* models/* .pytest_cache
	touch data/raw/.gitkeep data/processed/.gitkeep models/.gitkeep
