PYTHON ?= python3

.PHONY: all run test data workstreams timeseries rolling dixon_coles notebook clean

# Core pipeline (about 25 s) followed by the test suite.
all: run test

run:
	$(PYTHON) run_all.py

test:
	$(PYTHON) -m pytest -q tests

data:
	$(PYTHON) -c "from src.data import build_matches, ALL_SEASONS; build_matches(ALL_SEASONS)"

# The three workstream scripts (each under 15 minutes on a laptop; about 7, 10 and 1 minutes).
# Run them before `make run` so results/SUMMARY.md quotes their headline numbers.
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

clean:
	rm -rf results/*.json results/*.csv results/SUMMARY.md figures/*.png figures/*.pdf data/processed/*.csv
