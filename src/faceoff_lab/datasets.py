"""Built-in dataset loaders and a small interface for custom adapters."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .project import ProjectConfig, SourceConfig

DOCUMENT_SUFFIXES = {".md", ".txt", ".pdf"}


@dataclass
class SourceRecord:
    """A retrievable unit before document chunking."""

    id: str
    source: str
    text: str
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


RecordLoader = Callable[[ProjectConfig, SourceConfig], Iterable[SourceRecord]]


def _display_path(project: ProjectConfig, path: Path) -> str:
    try:
        return str(path.relative_to(project.root))
    except ValueError:
        return str(path)


def _document_text(path: Path) -> str:
    if path.suffix.lower() != ".pdf":
        return path.read_text(encoding="utf-8-sig")
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support requires the 'pdf' extra: pip install -e '.[pdf]'") from exc
    reader = PdfReader(str(path))
    pages = []
    for number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {number}]\n{text.strip()}")
    if not pages:
        raise ValueError(f"No extractable text found in {path.name}. Scanned PDFs need OCR first.")
    return "\n\n".join(pages)


def load_documents(project: ProjectConfig, source: SourceConfig) -> Iterable[SourceRecord]:
    root = project.resolve(source.path)
    if root.is_file():
        paths = [root]
    elif root.is_dir():
        paths = sorted(
            path
            for path in root.glob(source.glob)
            if path.is_file()
            and path.suffix.lower() in DOCUMENT_SUFFIXES
            and not any(part.startswith(".") for part in path.relative_to(root).parts)
        )
    else:
        raise FileNotFoundError(f"Document source not found: {root}")
    if not paths:
        raise ValueError(f"No .md, .txt, or .pdf files found in {root}")
    for path in paths:
        text = _document_text(path).strip()
        if not text:
            continue
        relative = _display_path(project, path)
        yield SourceRecord(
            id=relative,
            source=relative,
            text=text,
            kind="document",
            metadata={"filename": path.name, "suffix": path.suffix.lower()},
        )


def _row_values(row: dict[str, Any], columns: list[str], source: SourceConfig) -> str:
    missing = [column for column in columns if column not in row]
    if missing:
        raise ValueError(f"{source.path}: missing configured columns: {', '.join(missing)}")
    return "\n".join(f"{column}: {row.get(column) or ''}" for column in columns).strip()


def _structured_record(
    row: dict[str, Any], source: SourceConfig, source_name: str, row_number: int
) -> SourceRecord | None:
    text = _row_values(row, source.text_columns, source)
    if not text or not any(str(row.get(column) or "").strip() for column in source.text_columns):
        return None
    record_id = str(row.get(source.id_column) or f"{source_name}:row-{row_number}")
    metadata_columns = source.metadata_columns or [
        column for column in row if column not in source.text_columns and column != source.id_column
    ]
    metadata = {column: row.get(column) for column in metadata_columns if column in row}
    metadata["row"] = row_number
    return SourceRecord(
        id=record_id,
        source=source_name,
        text=text,
        kind=source.kind,
        metadata=metadata,
    )


def load_csv(project: ProjectConfig, source: SourceConfig) -> Iterable[SourceRecord]:
    path = project.resolve(source.path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV source not found: {path}")
    source_name = _display_path(project, path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        for row_number, row in enumerate(reader, 1):
            record = _structured_record(row, source, source_name, row_number)
            if record:
                yield record


def load_jsonl(project: ProjectConfig, source: SourceConfig) -> Iterable[SourceRecord]:
    path = project.resolve(source.path)
    if not path.is_file():
        raise FileNotFoundError(f"JSONL source not found: {path}")
    source_name = _display_path(project, path)
    with path.open(encoding="utf-8-sig") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{row_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{row_number}: every JSONL row must be an object.")
            record = _structured_record(row, source, source_name, row_number)
            if record:
                yield record


LOADERS: dict[str, RecordLoader] = {
    "documents": load_documents,
    "csv": load_csv,
    "jsonl": load_jsonl,
}


def register_loader(kind: str, loader: RecordLoader) -> None:
    """Register a custom adapter without modifying the retrieval engine."""
    if not kind or kind in LOADERS:
        raise ValueError(f"Loader kind is empty or already registered: {kind}")
    LOADERS[kind] = loader


def load_records(project: ProjectConfig) -> list[SourceRecord]:
    project.validate()
    records: list[SourceRecord] = []
    seen: set[str] = set()
    for source in project.sources:
        loader = LOADERS.get(source.kind)
        if loader is None:
            raise ValueError(f"No loader registered for source kind: {source.kind}")
        for record in loader(project, source):
            if record.id in seen:
                raise ValueError(f"Duplicate record ID: {record.id}")
            seen.add(record.id)
            records.append(record)
    if not records:
        raise ValueError("The configured sources produced no non-empty records.")
    return records


def records_fingerprint(project: ProjectConfig, records: list[SourceRecord]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps([asdict(s) for s in project.sources], sort_keys=True, default=str).encode())
    for record in records:
        digest.update(json.dumps(record.to_dict(), sort_keys=True, default=str).encode())
    return digest.hexdigest()[:20]


def source_columns(path: str | Path, kind: str) -> list[str]:
    """Read only a header/sample for the dataset-mapping UI."""
    if kind == "csv":
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle).fieldnames or [])
    if kind == "jsonl":
        with Path(path).open(encoding="utf-8-sig") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("JSONL rows must be objects.")
                    return list(value)
        return []
    raise ValueError("Column discovery is only available for CSV and JSONL.")
