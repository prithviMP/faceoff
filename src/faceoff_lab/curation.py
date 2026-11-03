"""Draft creation and explicit human review. Drafts never claim approval."""

from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import os
import yaml
from .datasets import SourceRecord
from .okf import load_page


def write_page(path: Path, label: dict, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n" + yaml.safe_dump(label, sort_keys=False, allow_unicode=True) + "---\n" + body,
        encoding="utf-8",
    )


def draft_bundle(records: list[SourceRecord], destination: Path, source_root: Path | None = None) -> Path:
    if destination.exists():
        raise ValueError("Draft destination already exists; choose a new folder.")
    destination.mkdir(parents=True)
    links = []
    for i, record in enumerate(records, 1):
        name = f"page-{i:04d}.md"
        title = record.metadata.get("filename") or record.id
        resource = record.source
        if source_root is not None and (source_root / record.source).is_file():
            resource = os.path.relpath((source_root / record.source).resolve(), destination.resolve())
        write_page(
            destination / name,
            {
                "type": "Reference",
                "title": str(title),
                "description": f"Draft extracted from {record.source}; edit this description before navigating.",
                "status": "draft",
                "verified": [],
                "sources": [{"id": "original", "resource": resource, "title": record.id}],
            },
            record.text,
        )
        links.append(f"[{title}]({name}) — Draft from {record.source}; edit the topic description.")
    (destination / "index.md").write_text("# Contents\n\n" + "\n".join(links), encoding="utf-8")
    return destination


def mark_reviewed(path: Path, reviewer: str) -> None:
    reviewer = reviewer.strip()
    if not reviewer or any(c in reviewer for c in "\n\r:"):
        raise ValueError("Enter your reviewer ID without colons or newlines.")
    page = load_page(path)
    if not page.label.get("type"):
        raise ValueError("Set a page type before recording review.")
    label = dict(page.label)
    prior = label.get("verified") or []
    if isinstance(prior, dict):
        prior = [prior]
    if not isinstance(prior, list):
        raise ValueError("verified must be a list of review entries.")
    label["verified"] = prior + [{"by": f"human:{reviewer}", "at": datetime.now(timezone.utc).isoformat()}]
    write_page(path, label, page.body)
