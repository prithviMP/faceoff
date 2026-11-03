# Contributing

Use Python 3.11+, create a virtual environment and install `python -m pip install -e '.[dev,pdf]'`.
Run `python -m ruff check src tests` and `python -m pytest -q` before opening a PR.

Keep source adapters separate from engines. New adapters yield SourceRecord objects with stable unique IDs, exact text and provenance metadata. Include tests for a dataset unrelated to the built-in examples. Optional features must fail clearly when an extra/model/provider is absent. Use test doubles for provider contract tests, and real provider adapters for application generation.

Do not commit imported private files, credentials, `.faceoff` caches, model weights, generated run logs or personal data. Include source licenses for sample datasets. Make small focused changes; explain the behavior and validation in the PR. Add a runnable workshop checkpoint for new student-facing capabilities. Release checks include wheel-installed examples from a clean working directory.
