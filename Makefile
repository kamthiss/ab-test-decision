PY = .venv/bin/python

.PHONY: setup design data check test notebook

setup:
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
