PYTHON ?= python3

.PHONY: all run test data workstreams timeseries rolling dixon_coles notebook reproduce clean

# Core pipeline (about 25 s) followed by the test suite.
all: run test

run:
	$(PYTHON) run_all.py

test:
	$(PYTHON) -m pytest -q tests

data:
	$(PYTHON) -c "from src.data import build_matches, ALL_SEASONS; build_matches(ALL_SEASONS)"

# The three workstream scripts (each under 15 minutes on a laptop; about 10, 15 and 2 minutes).
# They read results/numbers.json and results/home_advantage_by_season.csv, so run `make run` first;
# run `make run` again afterwards so results/SUMMARY.md quotes their headline numbers.
workstreams: timeseries rolling dixon_coles

timeseries:
	$(PYTHON) scripts/run_timeseries.py

rolling:
	$(PYTHON) scripts/run_rolling.py

dixon_coles:
	$(PYTHON) scripts/run_dixon_coles.py

# Execute the report notebook from a clean kernel, in place (about 10 s once the results files exist).
notebook:
	$(PYTHON) -m nbconvert --to notebook --execute --inplace notebooks/final_report.ipynb --ExecutePreprocessor.timeout=600

# Everything from a fresh clone, in the order that works (about 30 minutes).
reproduce:
	$(MAKE) run
	$(MAKE) workstreams
	$(MAKE) all
	$(MAKE) notebook

clean:
	rm -rf results/*.json results/*.csv results/SUMMARY.md figures/*.png figures/*.pdf data/processed/*.csv
