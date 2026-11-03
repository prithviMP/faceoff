"""Shared execution records. No templates, winners, date tricks or implicit grading."""

from __future__ import annotations
import json
import re
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from .datasets import load_records
from .llm import LLMSettings, create_generator, answer_from_context, resolve_settings
from .okf import OKFReader
from .project import ProjectConfig
from .rag import RAGIndex


def run(
    project: ProjectConfig,
    system: str,
    question: str,
    settings: LLMSettings | None = None,
    index: RAGIndex | None = None,
    parameters: dict | None = None,
) -> dict:
    settings = resolve_settings(settings or LLMSettings())
    start = time.perf_counter()
    result = {
        "system": system,
        "status": "error",
        "answer": None,
        "reason": "",
        "sources": [],
        "trace": {},
        "words_read": 0,
    }
    try:
        if not question.strip():
            raise ValueError("Enter a question.")
        generator = create_generator(settings)
        if system == "RAG":
            index = index or RAGIndex(project, load_records(project))
            sources, trace = index.search(question)
            result.update(
                sources=sources, trace=trace, words_read=sum(len(s["text"].split()) for s in sources)
            )
            if not sources:
                result.update(
                    status="insufficient_evidence", reason="No matching source passages were found."
                )
            elif not generator:
                result.update(
                    status="evidence",
                    reason="Answer generation is off. Inspect the retrieved passages below.",
                )
            else:
                result.update(status="answered", answer=answer_from_context(question, sources, generator))
        elif system == "OKF":
            if not project.okf_path:
                result.update(
                    status="no_coverage", reason="No handbook bundle is configured for this dataset."
                )
            else:
                result = OKFReader(project, generator).answer(question, computation_parameters=parameters)
        else:
            raise ValueError("Choose RAG or OKF.")
    except Exception as exc:
        result.update(status="error", reason=str(exc), answer=None)
    if (result.get("answer") or "").startswith("INSUFFICIENT_EVIDENCE:"):
        result.update(
            status="insufficient_evidence", reason=result["answer"].split(":", 1)[1].strip(), answer=None
        )
    result.update(
        run_id=uuid.uuid4().hex,
        question=question,
        project=project.name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        provider=asdict(settings),
        retrieval_settings=asdict(project.rag),
        latency_seconds=time.perf_counter() - start,
    )
    # Each citation is mapped to an exact admitted source passage, never a file guessed by the model.
    numbers = [int(n) for n in re.findall(r"\[(\d+)\]", result.get("answer") or "")]
    result["citations"] = [
        {"number": n, "source_id": result["sources"][n - 1]["id"]}
        for n in sorted(set(numbers))
        if 1 <= n <= len(result["sources"])
    ]
    invalid = sorted({n for n in numbers if n < 1 or n > len(result["sources"])})
    result["citation_warnings"] = [f"Unknown citation [{n}]" for n in invalid]
    if result.get("answer") and not numbers:
        result["citation_warnings"].append(
            "Answer has no source citations; verify its claims against the evidence."
        )
    project.workspace_path.mkdir(parents=True, exist_ok=True)
    with (project.workspace_path / "runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
    return result


def compare_traces(results: dict) -> list[str]:
    rag, okf = results.get("RAG"), results.get("OKF")
    if not rag or not okf:
        return ["Run both systems to compare their actual evidence."]
    observations = [
        f"RAG selected {len(rag['sources'])} passages; OKF opened {len(okf['trace'].get('trail', []))} pages."
    ]
    observations.append(f"RAG state: {rag['status']}; OKF state: {okf['status']}.")
    if okf["trace"].get("checks"):
        checks = okf["trace"]["checks"]
        observations.append(
            f"Handbook review: {checks['trust']}; current: {checks['current']}; within deadline: {checks['valid']}."
        )
    if okf["trace"].get("inspector"):
        observations.append(
            "Computation check: " + ("passed" if okf["trace"]["inspector"]["passed"] else "failed") + "."
        )
    return observations


def evaluate(result: dict, expectations: dict, question: str) -> dict:
    """Optional literal checks from an explicitly supplied answer key, not a correctness oracle."""
    expected = expectations.get(question)
    if expected is None:
        return {"status": "ungraded", "reason": "No expectation was supplied for this question."}
    checks = []
    if "status" in expected:
        checks.append(result["status"] == expected["status"])
    for phrase in expected.get("contains", []):
        checks.append(str(phrase).casefold() in (result.get("answer") or "").casefold())
    return {
        "status": "pass" if checks and all(checks) else "fail" if checks else "ungraded",
        "reason": "Literal checks against your supplied expectations; this is not semantic grading.",
    }


def load_expectations(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Answer key must be a JSON mapping of questions to checks.")
    return value
