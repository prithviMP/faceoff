# Add real embeddings and rank fusion

## Learning objective

Add real embeddings and rank fusion, and explain which evidence establishes the observed behavior.

## Source files

`src/faceoff_lab/rag.py: prepare, _load_encoder, search` (paths after commas refer to the same package unless otherwise stated).

## Small implementation steps

Install the semantic extra. Set rag.semantic to true in project.yaml or use Build RAG. Prepare the semantic index once, then compare a paraphrase with the keyword-only result. Inspect both raw scores and RRF ranking.

## Runnable checkpoint

```sh
python -m pip install -e ".[semantic]"
faceoff index workspace/lesson
faceoff query workspace/lesson "broken merchandise return window" --system RAG
```

## Evidence to inspect

Confirm embeddings.npy and index.json belong to the current content fingerprint. Similarity can find paraphrases but does not check authority. No outcome is predetermined.

## Common mistakes and troubleshooting

Changing piece boundaries needs new vectors. Changing top-k does not. Model downloads need network once; use a prepared local folder for disconnected workshops. Optional dependencies are not in the lightweight baseline.
