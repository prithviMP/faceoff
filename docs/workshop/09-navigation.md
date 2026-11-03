# Implement contents-first navigation and evidence admission

## Learning objective

Implement contents-first navigation and evidence admission, and explain which evidence establishes the observed behavior.

## Source files

`src/faceoff_lab/okf.py: OKFReader, deterministic_choice` (paths after commas refer to the same package unless otherwise stated).

## Small implementation steps

Read how index choices are scored without a provider. Ask a question matching an edited description. With a provider, inspect the returned link and any fallback. Follow the trail, then check which page body was admitted.

## Runnable checkpoint

```sh
faceoff query workspace/lesson "damaged return policy" --system OKF
```

## Evidence to inspect

Inspect menu descriptions, chosen links, review checks and admitted content. A no-coverage result identifies a gap in the handbook. Ties can require clarification.

## Common mistakes and troubleshooting

Deterministic lexical navigation is conservative and not a general semantic router. A broad root description can hide a useful topic. Bad descriptions and malformed links should be fixed visibly.
