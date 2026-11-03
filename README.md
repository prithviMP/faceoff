# Faceoff Lab

Build a retrieval assistant and an OKF handbook reader on your own dataset, then inspect what each actually read. This is a local Streamlit + Python learning workbench. **OKF is a knowledge format; Faceoff Lab supplies a reference runtime.**

## Quickstart · no model, key or cached data required

Python 3.11 or newer:

```sh
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
python -m pip install -e .
faceoff example policies workspace/policies
faceoff inspect workspace/policies
faceoff index workspace/policies
faceoff query workspace/policies "damaged return policy"
faceoff ui
```

Windows PowerShell: use `py -3.11 -m venv .venv`, then `.venv\Scripts\Activate.ps1`. If activation is unavailable, run `.venv\Scripts\python -m pip install -e .` and `.venv\Scripts\python -m faceoff_lab ui` directly; no `make` or shell activation is required. On macOS/Linux the equivalent direct executable is `.venv/bin/python`.

Open **http://localhost:8503**. Choose **Open project** and enter `workspace/policies`, or start a fresh example in a new working folder. The initial mode shows actual source passages, rankings, navigation and checks. Answer generation is disabled until you choose a provider. The optional reviewed SQLite example can compute a real result without an LLM.

The UI steps are **Inspect data → Build RAG → Curate OKF → Compare**. Import a local folder, supported file, or ZIP from the welcome screen. Set CSV/JSONL column mappings before querying. In Compare, ask one question and choose Run RAG, Run OKF, or Run both. Citation numbers link to the admitted passage. Export project settings or complete run traces from the UI.

## Real answer providers and optional features

Install only what you need:

```sh
python -m pip install -e '.[gemini]'     # Gemini SDK
python -m pip install -e '.[semantic]'   # CPU embeddings and optional reranker
python -m pip install -e '.[pdf]'        # PDFs with extractable text
python -m pip install -e '.[dev]'        # tests and lint
```

Copy `.env.example` to `.env` and set `GEMINI_API_KEY` for Gemini. In Compare select Gemini and your model ID, or run:

```sh
faceoff query workspace/policies "damaged return policy" --provider gemini --model YOUR_MODEL_ID
```

Ollama uses its real HTTP endpoint (no Python provider extra): install Ollama separately, pull the model you want and select its actual model name, for example `llama3.2:3b`, with endpoint `http://localhost:11434`. OpenAI-compatible providers require a model ID and base URL, including `/v1` where required; `OPENAI_API_KEY` is read from the environment. Keys are never exported. Network/provider failures appear as errors with their provider details. Paid or local model quality and availability are not guaranteed by this project.

Keyword search is BM25+. It works on fresh installations without downloads. Meaning search adds real sentence-transformer vectors and reciprocal rank fusion. Enable it in Build RAG, choose a model or local model folder, then press Prepare semantic index. Changing source content, piece boundaries or the embedding model creates a different cache. Changing top-k does not invalidate embeddings. The optional second reader runs a cross-encoder over the best candidates. The first model use may download weights; use a local model folder or your provider's normal offline configuration for disconnected use.

## Examples

`faceoff example NAME NEW_FOLDER` copies data and configuration out of the installed package; examples are included in built wheels.

- **policies**: a clearly labelled synthetic MeraBazaar old/current rule pair with a fictional reviewed handbook. Inspect similarity versus lifecycle. Change top-k and observe; no answer is guaranteed.
- **support**: 30 synthetic English/Hinglish tickets. `faceoff generate-tickets tickets-large.csv --rows 10000` produces an optional scale exercise; configure its column mappings in a project. Retrieval samples are not exhaustive counts.
- **refunds**: synthetic SQLite ledger with a fictional, stable, reviewed recipe. Ask `refund total sale_id=DEMO` or supply `{"sale_id":"DEMO"}` in Compare. The result is in paise as specified by the recipe; no currency formatting or schema is assumed by the engine.
- **iris**: real public UCI Iris data, converted to searchable measurement descriptions. Includes license, attribution, transformation notes and questions. Source: [UCI Iris](https://archive.ics.uci.edu/dataset/53/iris), Fisher (1936), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Search is not statistical aggregation.

Each example includes `project.yaml`, source data and `questions.md`. The synthetic input data is educational; it is not used as a hidden answer key. All example review actors are explicitly fictional. Automatically drafted pages have no review actors.

## Bring your own dataset

```sh
faceoff import /path/to/documents workspace/research
faceoff inspect workspace/research
faceoff index workspace/research
faceoff query workspace/research "your question"
faceoff draft workspace/research
faceoff validate workspace/research
```

Built-in loaders: Markdown, plain text, extractable PDF, CSV and JSONL. ZIPs are bounded and reject path traversal and symbolic links. Folders can contain mixed supported files. JSONL rows must be objects; CSV has a header. IDs must be unique across sources. The UI suggests text fields, but you should inspect and change the mapping. Scanned PDFs need external OCR; DOCX, images and arbitrary binary formats are not automatically supported.

This is **any dataset through an adapter, with common formats supported out of the box**, not a universal parser. See [architecture and adapter interface](docs/ARCHITECTURE.md).

A project configuration records source paths/mappings, piece settings, optional models, handbook root and workspace location. Relative paths are resolved from that config, not from the current terminal directory. Exporting the configuration does not bundle your source files. Share the project folder with its sources, `project.yaml` and handbook; omit `.faceoff`, keys and private logs. Absolute external paths must be made portable by the author.

Before sharing a dataset, check its license, authority to use it, personal/private information and provenance. Local imports stay local until you choose a remote provider: **selected evidence and index descriptions are then sent to that endpoint**. Prefer licensed public or synthetic data for workshops.

## Curation, computations and honest states

Drafting creates one unverified reference page per selected record and a root index. It copies text; it does not invent trust, summaries, sources or SQL recipes. Edit descriptions, metadata and source links, validate the bundle, and explicitly record your review after checking the sources. Review declarations are local metadata, not authenticated signatures. Updating review does not silently change draft status or extend an expiry date.

OKF navigation reads index descriptions. Without an LLM, conservative lexical navigation may have no match or a tie; clarify the question or edit descriptions. With a provider, the provider selects an existing link, with a logged fallback for malformed output. The reader refuses deprecated or expired content before admission. Unverified content is visibly warned about. Computation execution additionally requires human review and a supported SQLite runtime; it never executes an automatically generated recipe.

Recipes use named bound parameters. The source SQLite connection is read-only. A separate local receipt store records SQL, parameters, source and result. Deterministic inspection reloads the execution record and independently rereads the recipe; a mismatch withholds the result. This illustrates a contract, not adversarial cryptographic attestation. Only trusted, reviewed local recipes should be used; the app is not a public SQL service.

There is no scoreboard and no automatic Right/Wrong. If you supply a JSON answer key, literal phrase/status checks are displayed separately as **your evaluation**. These are not semantic correctness judgments. `examples` contain no built-in grading file.

Each run records provider/model, actual settings, question, state, latency, exact source IDs/passages, navigation/selection details, citations and optional receipt. A citation's existence is validated; whether its passage logically supports the model's claim still needs inspection. Timings include engine/provider work and cold model loading, not UI rendering. Run both executes each pipeline in order; it is an experiment, not a universal speed benchmark. Dataset/curation changes clear visible results; runs are snapshots and new provider/question controls apply on the next run.

## Learn to build it

Start at [docs/workshop/README.md](docs/workshop/README.md). Twelve checkpoints take you from a raw record through chunking, BM25, vectors, real providers, handbook curation, receipt checks and custom datasets. Each chapter includes source files, steps, a runnable check, evidence to inspect and common mistakes.

## Development

```sh
python -m pip install -e '.[dev,pdf]'
python -m ruff check src tests
python -m pytest -q
```

CI repeats those commands on Linux and Windows, Python 3.11/3.12, then builds a wheel and exercises its retrieval-only quickstart. External model/network calls are not required by CI; test doubles are confined to tests. Optional real provider smoke tests need your own endpoint/key. See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md) and [limitations](docs/ARCHITECTURE.md).

