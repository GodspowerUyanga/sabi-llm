# SABI - common developer tasks
# Usage: make <target>

PYTHON ?= python3
VENV   ?= .venv
PIP     = $(VENV)/bin/pip
PY      = $(VENV)/bin/python

.PHONY: help venv install install-dev model run chat doctor benchmark profile test lint clean

help:
	@echo "SABI make targets:"
	@echo "  venv         create a virtual environment in $(VENV)"
	@echo "  install      install runtime dependencies + the sabi package"
	@echo "  install-dev  install dev dependencies (tests, lint)"
	@echo "  model        download the GGUF model from Hugging Face"
	@echo "  run          start the SABI runtime"
	@echo "  chat         launch the chat UI"
	@echo "  doctor       diagnose the environment"
	@echo "  benchmark    run the local benchmark"
	@echo "  profile      print RAM/CPU/thermal telemetry"
	@echo "  test         run the test suite"
	@echo "  lint         run ruff"
	@echo "  clean        remove caches and build artifacts"

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev: venv
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e ".[dev]"

model:
	$(PY) scripts/download_model.py

run:
	$(PY) -m sabi run

chat:
	$(PY) -m sabi chat

doctor:
	$(PY) -m sabi doctor

benchmark:
	$(PY) scripts/run_benchmark.py

profile:
	$(PY) -m sabi profile

test:
	$(PY) -m pytest

lint:
	$(VENV)/bin/ruff check sabi tests scripts

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
