# Implement and inspect character splitting

## Learning objective

Implement and inspect character splitting, and explain which evidence establishes the observed behavior.

## Source files

`src/faceoff_lab/rag.py: chunk_records, tests/test_lab.py` (paths after commas refer to the same package unless otherwise stated).

## Small implementation steps

Read chunk_records. Follow start/end offsets for one record. Change chunk_size and overlap in the project configuration, keeping overlap smaller than piece size. Run the index checkpoint. Add a test asserting each slice equals record.text[start:end].

## Runnable checkpoint

```sh
faceoff index workspace/lesson
python -m pytest tests/test_lab.py -q
```

## Evidence to inspect

Compare overlapping pieces in the Build RAG view. Identify whether an exception or sentence is cut. Smaller pieces change evidence boundaries, not truth.

## Common mistakes and troubleshooting

Overlap equal to size never advances the cut. A boundary mid-rule may leave selected context incomplete. Keep offsets exact when writing a new splitter.
