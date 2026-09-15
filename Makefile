.PHONY: install sample test evaluate lint clean

install:
	pip install -e ".[dev]"

sample:
	python scripts/make_sample.py

test:
	python -m pytest -q

evaluate:
	python scripts/evaluate.py --data data/sample

lint:
	ruff check src tests scripts

clean:
	rm -rf results data/sample .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
