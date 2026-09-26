PYTHON ?= python3

.PHONY: all run test data clean

all: run test

run:
	$(PYTHON) run_all.py

test:
	$(PYTHON) -m pytest -q tests

data:
	$(PYTHON) -c "from src.data import build_matches, ALL_SEASONS; build_matches(ALL_SEASONS)"

clean:
	rm -rf results/*.json results/*.csv results/SUMMARY.md figures/*.png figures/*.pdf data/processed/*.csv
