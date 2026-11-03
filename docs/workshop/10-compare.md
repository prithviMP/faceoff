# Run both systems on the same question

## Learning objective

Run both systems on the same question, and explain which evidence establishes the observed behavior.

## Source files

`src/faceoff_lab/engine.py: run, compare_traces; ui.py: compare_step` (paths after commas refer to the same package unless otherwise stated).

## Small implementation steps

Open Compare and use Run both. Follow selected context and handbook choices. Change top-k, rerun, and compare saved snapshots. Export the project configuration and each complete run trace.

## Runnable checkpoint

```sh
faceoff query workspace/lesson "damaged return policy" --system both
```

## Evidence to inspect

Explain differences using actual source IDs, admission decisions and missing coverage. Timings include engine work and possible cold model loading. No scoreboard or fixed winner exists.

## Common mistakes and troubleshooting

Do not treat one latency measurement as a general benchmark. Changed controls apply to the next run. When sharing settings, also include source files and handbook paths, not private .faceoff logs.
