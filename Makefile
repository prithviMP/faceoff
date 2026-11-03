PY = .venv/bin/python
setup:
	python3 -m venv .venv
	$(PY) -m pip install -e '.[dev]'
faceoff:
	$(PY) -m faceoff_lab ui
rag: faceoff
okf: faceoff
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check src tests
