"""Portable project configuration for datasets, retrieval, and OKF bundles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_NAMES = ("faceoff.yaml", "faceoff.yml", "project.yaml", "project.yml")


@dataclass
class SourceConfig:
    """One document collection or structured data source."""

    kind: str
    path: str
    glob: str = "**/*"
    id_column: str = "id"
    text_columns: list[str] = field(default_factory=lambda: ["text"])
    metadata_columns: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceConfig":
        if not isinstance(value, dict):
            raise ValueError("Each source must be a YAML mapping.")
        try:
            return cls(**value)
        except TypeError as exc:
            raise ValueError(f"Invalid source configuration: {exc}") from exc


@dataclass
class RAGConfig:
    """Reproducible retrieval settings; no answer or grading information."""

    chunk_size: int = 300
    overlap: int = 40
    top_k: int = 3
    semantic: bool = False
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    rerank: bool = False
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    def validate(self) -> None:
        if self.chunk_size < 50:
            raise ValueError("Chunk size must be at least 50 characters.")
        if not 0 <= self.overlap < self.chunk_size:
            raise ValueError("Overlap must be non-negative and smaller than chunk size.")
        if not 1 <= self.top_k <= 50:
            raise ValueError("Top-k must be between 1 and 50.")


@dataclass
class ProjectConfig:
    """A complete, shareable experiment rooted at its configuration file."""

    name: str
    sources: list[SourceConfig]
    description: str = ""
    rag: RAGConfig = field(default_factory=RAGConfig)
    okf_root: str | None = None
    workspace: str = ".faceoff"
    root: Path = field(default_factory=Path.cwd, repr=False, compare=False)

    @classmethod
    def from_dict(cls, value: dict[str, Any], root: Path) -> "ProjectConfig":
        if not value.get("name"):
            raise ValueError("Project configuration needs a name.")
        sources = [SourceConfig.from_dict(item) for item in value.get("sources", [])]
        if not sources:
            raise ValueError("Project configuration needs at least one source.")
        rag_value = value.get("rag", {})
        rag = RAGConfig(**rag_value) if isinstance(rag_value, dict) else RAGConfig()
        project = cls(
            name=str(value["name"]),
            description=str(value.get("description", "")),
            sources=sources,
            rag=rag,
            okf_root=value.get("okf_root"),
            workspace=str(value.get("workspace", ".faceoff")),
            root=root,
        )
        project.validate()
        return project

    def validate(self) -> None:
        self.rag.validate()
        for source in self.sources:
            from .datasets import LOADERS

            if source.kind not in LOADERS:
                raise ValueError(f"Unsupported source kind: {source.kind}")
            if not source.path:
                raise ValueError("Every source needs a path.")
            if source.kind != "documents" and not source.text_columns:
                raise ValueError(f"{source.path}: select at least one text column.")

    def resolve(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()

    @property
    def okf_path(self) -> Path | None:
        return self.resolve(self.okf_root) if self.okf_root else None

    @property
    def workspace_path(self) -> Path:
        return self.resolve(self.workspace)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "sources": [asdict(source) for source in self.sources],
            "rag": asdict(self.rag),
            "okf_root": self.okf_root,
            "workspace": self.workspace,
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return destination


def config_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        for name in CONFIG_NAMES:
            if (candidate / name).is_file():
                return candidate / name
        raise FileNotFoundError(
            f"No project configuration in {candidate}. Expected one of: {', '.join(CONFIG_NAMES)}"
        )
    return candidate


def load_project(path: str | Path) -> ProjectConfig:
    configuration = config_path(path)
    if not configuration.is_file():
        raise FileNotFoundError(f"Project configuration not found: {configuration}")
    value = yaml.safe_load(configuration.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("Project configuration must be a YAML mapping.")
    return ProjectConfig.from_dict(value, configuration.parent)


def discover_projects(root: str | Path) -> list[Path]:
    base = Path(root)
    if not base.exists():
        return []
    return sorted(path for path in base.glob("*/project.yaml") if path.is_file())
