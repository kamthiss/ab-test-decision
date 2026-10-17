PY = .venv/bin/python

.PHONY: setup design data check test

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
