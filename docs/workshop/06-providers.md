# Connect a real answer model and require citations

## Learning objective

Connect a real answer model and require citations, and explain which evidence establishes the observed behavior.

## Source files

`src/faceoff_lab/llm.py, engine.py` (paths after commas refer to the same package unless otherwise stated).

## Small implementation steps

Select Gemini, a running Ollama model or an OpenAI-compatible endpoint. Enter the actual model name. Read answer_from_context and inspect the evidence passed to the provider. Follow each returned citation to its source passage.

## Runnable checkpoint

```sh
faceoff query workspace/lesson "damaged return policy" --provider ollama --model llama3.2:3b
```

## Evidence to inspect

Compare answer, citation map and original passage. The run records provider settings and latency. A model failure remains an error; none of the runtime paths writes a template answer.

## Common mistakes and troubleshooting

A valid citation number does not prove the passage supports the claim. Inspect entailment yourself. A missing API key or unloaded local model is not a retrieval failure. Never commit keys.
