"""Local imports with bounded ZIP extraction; imported files are never executed."""

from __future__ import annotations
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path
from .datasets import source_columns
from .project import ProjectConfig, SourceConfig, load_project

MAX_BYTES = 100 * 1024 * 1024
MAX_FILES = 2000


def unpack_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        if len(members) > MAX_FILES or sum(m.file_size for m in members) > MAX_BYTES:
            raise ValueError("Archive exceeds 2,000 files or 100 MB uncompressed.")
        for m in members:
            path = Path(m.filename.replace("\\", "/"))
            target = (destination / path).resolve()
            if path.is_absolute() or not target.is_relative_to(destination.resolve()):
                raise ValueError("Archive contains an unsafe path.")
            if stat.S_ISLNK(m.external_attr >> 16):
                raise ValueError("Symbolic links are not supported in archives.")
        for m in members:
            target = destination / Path(m.filename.replace("\\", "/"))
            if m.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)


def _import_path(source: Path, destination: Path) -> ProjectConfig:
    source = source.expanduser().resolve()
    if destination.exists():
        raise ValueError("Choose a new project folder; imports never overwrite an existing project.")
    destination.mkdir(parents=True)
    data = destination / "data"
    if source.is_dir():
        shutil.copytree(
            source,
            data,
            ignore=shutil.ignore_patterns(".git", ".venv", ".faceoff", "__pycache__"),
            symlinks=False,
        )
    elif source.suffix.lower() == ".zip":
        data.mkdir()
        unpack_zip(source, data)
    elif source.is_file() and source.suffix.lower() in {".md", ".txt", ".pdf", ".csv", ".jsonl"}:
        data.mkdir()
        shutil.copy2(source, data / source.name)
    else:
        raise ValueError("Select a folder, ZIP, Markdown, text, PDF, CSV or JSONL file.")
    sources = []
    documents = [p for p in data.rglob("*") if p.suffix.lower() in {".md", ".txt", ".pdf"}]
    if documents:
        sources.append(SourceConfig("documents", "data"))
    for p in sorted(data.rglob("*")):
        if p.suffix.lower() in {".csv", ".jsonl"}:
            kind = p.suffix[1:].lower()
            columns = source_columns(p, kind)
            preferred = next(
                (c for c in ["text", "message", "body", "content", "description"] if c in columns), None
            )
            # This is a suggested mapping; the UI makes it editable before loading.
            texts = [preferred] if preferred else columns[-1:]
            sources.append(
                SourceConfig(
                    kind,
                    str(p.relative_to(destination)),
                    text_columns=texts,
                    metadata_columns=[c for c in columns if c not in texts and c != "id"],
                )
            )
    project = ProjectConfig(source.stem, sources, root=destination.resolve())
    project.validate()
    project.save(destination / "project.yaml")
    return project


def import_path(source: Path, destination: Path) -> ProjectConfig:
    source, destination = source.expanduser().resolve(), destination.expanduser().resolve()
    if destination.exists():
        raise ValueError("Choose a new project folder; imports never overwrite an existing project.")
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir():
        if destination.is_relative_to(source):
            raise ValueError("The project destination must be outside the source folder.")
        if any(p.is_symlink() for p in source.rglob("*")):
            raise ValueError("Folder imports cannot contain symbolic links.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".faceoff-import-", dir=destination.parent) as staging:
        staged = Path(staging) / "project"
        _import_path(source, staged)
        shutil.move(str(staged), str(destination))
    return load_project(destination)
