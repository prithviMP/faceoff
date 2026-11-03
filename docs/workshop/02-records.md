# Load a document and inspect its records

## Learning objective

Load a document and inspect its records, and explain which evidence establishes the observed behavior.

## Source files

`src/faceoff_lab/datasets.py, project.py` (paths after commas refer to the same package unless otherwise stated).

## Small implementation steps

Create a text file about a topic unrelated to the examples. Import it into a fresh folder. Inspect its SourceRecord fields. For a CSV, set id_column and text_columns in project.yaml, then inspect again.

## Runnable checkpoint

```sh
faceoff import my-notes.txt workspace/my-notes
faceoff inspect workspace/my-notes
```

## Evidence to inspect

Check the source path, ID, loaded text and metadata. CSV search text includes configured field labels. PDF offsets refer to extracted text, not file bytes.

## Common mistakes and troubleshooting

Missing fields, duplicate IDs and non-UTF-8 input are data errors. Fix the mapping or encoding rather than hiding the error.
