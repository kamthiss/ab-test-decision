PY = .venv/bin/python

.PHONY: setup design

setup:
	uv pip install --python $(PY) -r requirements.txt
	uv pip install --python $(PY) -e .

design:
	$(PY) scripts/design.py
