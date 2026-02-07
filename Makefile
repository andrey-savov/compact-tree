.PHONY: help install install-dev test coverage clean build upload-test upload

help:
	@echo "Available commands:"
	@echo "  make install       Install the package in editable mode"
	@echo "  make install-dev   Install with development dependencies"
	@echo "  make test          Run tests"
	@echo "  make coverage      Run tests with coverage report"
	@echo "  make clean         Remove build artifacts"
	@echo "  make build         Build distribution packages"
	@echo "  make upload-test   Upload to Test PyPI"
	@echo "  make upload        Upload to PyPI"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest test_compact_tree.py -v

coverage:
	pytest test_compact_tree.py -v --cov=compact_tree --cov-report=html --cov-report=term

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.ctree" -delete

build: clean
	python -m build

upload-test: build
	python -m twine upload --repository testpypi dist/*

upload: build
	python -m twine upload dist/*
