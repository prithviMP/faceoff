# Run the evidence-only baseline

## Learning objective

Run the evidence-only baseline, and explain which evidence establishes the observed behavior.

## Source files

`src/faceoff_lab/project.py, cli.py` (paths after commas refer to the same package unless otherwise stated).

## Small implementation steps

Create a Python 3.11 virtual environment and install the package. Copy the policies example into a new working folder. Run inspect before any model is configured.

## Runnable checkpoint

```sh
faceoff example policies workspace/lesson
faceoff inspect workspace/lesson
faceoff query workspace/lesson "damaged return policy"
```

## Evidence to inspect

Inspect record count, exact source text and evidence state. There should be no generated answer or correctness badge.

## Common mistakes and troubleshooting

Activating a different Python than the one used for installation causes missing modules. Use python -m faceoff_lab to remove command-path ambiguity.
