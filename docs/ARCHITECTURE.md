# Architecture, formats and limits

`project.py` resolves portable YAML settings. `datasets.py` converts each configured source into `SourceRecord(id, source, text, kind, metadata)`. `rag.py` creates exact character slices with offsets; BM25+ ranks lexical matches, optional CPU vectors add meaning search, reciprocal-rank fusion merges rankings, and an optional cross-encoder rereads candidates. The cache identity covers source records, mappings, piece boundaries and embedding model; query top-k is not an embedding input.

`curation.py` copies selected records to unverified pages. `okf.py` traverses local Markdown contents links and admits pages only after metadata/lifecycle checks. OKF is a format, not a search engine or runtime. Navigation is imperfect; metadata is a declaration, not a guarantee. This runtime accepts unverified reference content with warnings and refuses unreviewed computations.

`llm.py` exposes real Gemini, Ollama and OpenAI-compatible adapters through `Generator.generate(prompt)`. Generation-off returns evidence without an answer. `engine.py` captures states, selected passages, actual timings and citation mappings; optional evaluations use only a supplied answer key. `receipts.py` executes read-only SQLite and independently verifies recorded SQL/parameters/results. `ui.py` and `cli.py` share these engines.

## A custom adapter

```python
from faceoff_lab.datasets import SourceRecord, register_loader

def load_notes(project, source):
    yield SourceRecord("stable-unique-id", source.path, "actual text", "custom", {"owner": "team"})

register_loader("custom", load_notes)
# Then load a project whose source kind is custom in that Python process.
```

Registration is explicit in the importing process. The built-in CLI/UI do not discover or execute plugins from uploaded files; a developer can create a wrapper or add an adapter in source. Add tests, document encoding, provenance and stable ID behavior. A SourceRecord's text must be exactly the text that offsets refer to; CSV text is the configured field-name/value projection, PDF text is extracted text, not raw binary bytes.

## Supported inputs

Markdown and UTF-8 text; extractable PDF with optional pypdf; CSV with headers; JSONL objects; folders and bounded ZIP files containing those formats. Multi-source IDs must be unique. Unsupported files are ignored in document folders. Scanned PDFs, images, DOCX, arbitrary databases and authenticated SaaS require adapters; they are not promised out of the box.

## Deliberate boundaries

Evaluation uses student-supplied expectations. Review metadata requires explicit curation. Top-k results are samples, not exhaustive counts, and measured timings describe individual runs. Local receipt consistency is not remote attestation. Missing source fields, broken links and malformed labels are visible errors. Handbook descriptions may not cover a question; a model can produce unsupported claims despite grounded prompts. Citations identify passages but do not establish logical entailment. Verify your provider and model configuration before relying on an experiment.
