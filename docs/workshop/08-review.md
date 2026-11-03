# Enforce review, status and expiry checks

## Learning objective

Enforce review, status and expiry checks, and explain which evidence establishes the observed behavior.

## Source files

`src/faceoff_lab/okf.py: check_label, validate_bundle; curation.py: mark_reviewed` (paths after commas refer to the same package unless otherwise stated).

## Small implementation steps

Edit a page label in Curate OKF. Deliberately break a link and repair it. Set a review deadline, inspect its status and record your own review only after checking sources. Use tests for expiry/deprecation scenarios.

## Runnable checkpoint

```sh
faceoff validate workspace/lesson
python -m pytest tests/test_lab.py -k "bundle_validation or expiry" -q
```

## Evidence to inspect

Validation distinguishes structural errors from unverified/expired/deprecated warnings. The reader refuses inadmissible lifecycle states. Review is a local declaration; it is not an authenticated digital signature.

## Common mistakes and troubleshooting

A human entry does not automatically mean current or unexpired. Recording review does not extend a date or change draft to stable. Set lifecycle fields intentionally.
