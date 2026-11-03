# Import an unrelated dataset and reproduce an experiment

## Learning objective

Import an unrelated dataset and reproduce an experiment, and explain which evidence establishes the observed behavior.

## Source files

`src/faceoff_lab/importing.py, datasets.py, project.py` (paths after commas refer to the same package unless otherwise stated).

## Small implementation steps

Try the public Iris example or your own CSV/JSONL. Inspect the license and provenance, map the fields, build the index and ask a question. Create a small handbook reference only if a topic needs curation.

## Runnable checkpoint

```sh
faceoff example iris workspace/iris
faceoff inspect workspace/iris
faceoff query workspace/iris "Iris-setosa petal width"
```

## Evidence to inspect

Citations resolve to actual measurement records. A species search does not establish an average across all rows. An absent handbook produces no coverage until you curate one.

## Common mistakes and troubleshooting

Automatic parsing is not automatic semantic understanding. Scanned PDFs need OCR; other formats need an adapter. Check rights and personal/private information before enabling a remote provider.
