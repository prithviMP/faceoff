# Extend evaluation, reranking or computation receipts

## Learning objective

Extend evaluation, reranking or computation receipts, and explain which evidence establishes the observed behavior.

## Source files

`src/faceoff_lab/receipts.py, engine.py: evaluate; examples/refunds` (paths after commas refer to the same package unless otherwise stated).

## Small implementation steps

Choose an extension. Add a second reader and compare selected context; supply your own phrase/status evaluation key; or inspect the refunds recipe. For calculations use the approved SQL and bound parameters, then inspect the independently loaded execution record.

## Runnable checkpoint

```sh
faceoff example refunds workspace/refunds
faceoff query workspace/refunds "refund total sale_id=DEMO" --system OKF
python -m pytest tests/test_lab.py -k "receipt or computation" -q
```

## Evidence to inspect

Inspect the actual query, parameter values, result and verification checks. Test altered SQL or receipt values inside tests. An unverified computation must not execute. A supplied answer key is optional and its literal checks are not semantic grading.

## Common mistakes and troubleshooting

Human review concerns the definition; per-run verification concerns execution consistency. A local receipt does not defeat a hostile host. Never invent a reviewed SQL recipe from an arbitrary document or use a model answer as a financial ledger.
