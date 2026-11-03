# Create an OKF contents page and linked topics

## Learning objective

Create an OKF contents page and linked topics, and explain which evidence establishes the observed behavior.

## Source files

`src/faceoff_lab/curation.py, okf.py: load_page, page_links` (paths after commas refer to the same package unless otherwise stated).

## Small implementation steps

Use the draft command on a small document project. Open the generated index and one concept. Replace generic descriptions with precise topics. Follow local links and add source provenance. Drafts remain unverified.

## Runnable checkpoint

```sh
faceoff draft workspace/lesson
faceoff validate workspace/lesson
```

## Evidence to inspect

Open handbook-draft/index.md and page-0001.md. The label should have type, draft status and an empty verified list. No human review or SQL recipe is invented.

## Common mistakes and troubleshooting

A bundle is not an automatic conversion of raw data into trusted knowledge. Do not draft ten thousand tickets as an accidental substitute for curation. Use the UI to select a few intended records.
