# Add BM25 retrieval and inspect rankings

## Learning objective

Add BM25 retrieval and inspect rankings, and explain which evidence establishes the observed behavior.

## Source files

`src/faceoff_lab/rag.py: tokenize, RAGIndex.search` (paths after commas refer to the same package unless otherwise stated).

## Small implementation steps

Trace tokenization, BM25Plus scores and ranked candidates. Ask an exact phrase, a record identifier included in text, and a question using unrelated words. Inspect selected passages before generation.

## Runnable checkpoint

```sh
faceoff query workspace/lesson "return policy" --system RAG
```

## Evidence to inspect

The trace distinguishes candidates from selected context. No lexical match should produce insufficient evidence. BM25+ is a lexical ranking, not an approval or freshness check.

## Common mistakes and troubleshooting

An ID stored only as metadata is not automatically searchable text. Add the ID to text_columns if the experiment should search it. Frequent words and wording affect ranks.
