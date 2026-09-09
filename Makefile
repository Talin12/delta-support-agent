.PHONY: help setup data threads taxonomy golden baselines agent judge agreement all clean

VENV := .venv/bin
BRAND := Delta

help:
	@echo "make setup      - create venv, install deps"
	@echo "make data       - download twcs.csv (~493MB)"
	@echo "make threads    - reconstruct $(BRAND) conversation threads"
	@echo "make golden     - resample + relabel the golden set (already committed)"
	@echo "make baselines  - run trivial + simple baselines (NO API KEY NEEDED)"
	@echo "make agent      - run the full agent (needs an LLM key, or the committed cache)"
	@echo "make judge      - run the LLM judge over all systems"
	@echo "make agreement  - judge vs human agreement report"
	@echo "make all        - headline results end to end"

setup:
	python3.12 -m venv .venv
	$(VENV)/pip install -q --upgrade pip
	$(VENV)/pip install -q -r requirements.txt
	@echo "setup complete"

data:
	bash scripts/download_data.sh

threads:
	$(VENV)/python src/prep.py --brand $(BRAND)

taxonomy:
	$(VENV)/python src/discover.py --brand $(BRAND) --k 16 --sample 6000 --no-llm

golden:
	$(VENV)/python scripts/sample_golden.py --brand $(BRAND)
	$(VENV)/python scripts/apply_labels.py

baselines:
	$(VENV)/python eval/harness.py --brand $(BRAND) --systems trivial,simple --no-judge

agent:
	$(VENV)/python eval/harness.py --brand $(BRAND) --systems agent --no-judge

judge:
	$(VENV)/python eval/harness.py --brand $(BRAND) --systems trivial,simple,agent

agreement:
	$(VENV)/python eval/agreement.py

all: threads baselines judge agreement
	$(VENV)/python scripts/make_report_tables.py

clean:
	rm -rf artifacts/*.json artifacts/*.jsonl artifacts/*.csv
