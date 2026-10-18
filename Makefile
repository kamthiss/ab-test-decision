PY = .venv/bin/python

.PHONY: all setup design data check test notebook simulations

all: setup data check test notebook simulations

setup:
	test -d .venv || uv venv --python 3.12 .venv
	uv pip install --python $(PY) -r requirements.txt
	uv pip install --python $(PY) -e .

design:
	$(PY) scripts/design.py

data:
	$(PY) -m generator.generate

check:
	$(PY) scripts/check_data.py

test:
	$(PY) -m pytest -q

notebook:
	.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb

simulations:
	.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/simulations.ipynb
