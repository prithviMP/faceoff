"""Transparent RAG indexing: character chunks, BM25, optional vectors, and RRF."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from rank_bm25 import BM25Plus

from .datasets import SourceRecord, records_fingerprint
from .project import ProjectConfig, RAGConfig

TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class Chunk:
    id: str
    record_id: str
    source: str
    text: str
    kind: str
    start: int
    end: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def chunk_records(records: list[SourceRecord], settings: RAGConfig) -> list[Chunk]:
    settings.validate()
    chunks: list[Chunk] = []
    stride = settings.chunk_size - settings.overlap
    for record in records:
        for start in range(0, len(record.text), stride):
            end = min(len(record.text), start + settings.chunk_size)
            text = record.text[start:end]
            if text.strip():
                chunks.append(
                    Chunk(
                        id=f"{record.id}::{start}",
                        record_id=record.id,
                        source=record.source,
                        text=text,
                        kind=record.kind,
                        start=start,
                        end=end,
                        metadata=record.metadata,
                    )
                )
            if end == len(record.text):
                break
    return chunks


class RAGIndex:
    """An index tied to one project and its exact content/configuration."""

    def __init__(self, project: ProjectConfig, records: list[SourceRecord]):
        self.project = project
        self.settings = project.rag
        self.records = records
        self.chunks = chunk_records(records, self.settings)
        if not self.chunks:
            raise ValueError("No searchable chunks were produced.")
        self.bm25 = BM25Plus([tokenize(chunk.text) or ["__empty__"] for chunk in self.chunks])
        self.fingerprint = self._fingerprint(records)
        self.index_dir = self.project.workspace_path / "indexes" / self.fingerprint
        self.vectors_path = self.index_dir / "embeddings.npy"
        self._encoder = None
        self._reranker = None

    def _fingerprint(self, records: list[SourceRecord]) -> str:
        value = {
            "records": records_fingerprint(self.project, records),
            "chunk_size": self.settings.chunk_size,
            "overlap": self.settings.overlap,
            "embedding_model": self.settings.embedding_model if self.settings.semantic else None,
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]

    @property
    def embeddings_ready(self) -> bool:
        if not self.vectors_path.is_file():
            return False
        try:
            vectors = np.load(self.vectors_path, mmap_mode="r")
            return vectors.shape[0] == len(self.chunks)
        except (ValueError, OSError):
            return False

    def _load_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Semantic search requires the 'semantic' extra: pip install -e '.[semantic]'"
                ) from exc
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self._encoder = SentenceTransformer(
                self.settings.embedding_model,
                device="cpu",
                cache_folder=str(self.project.workspace_path / "models"),
            )
        return self._encoder

    def prepare(self, progress: Callable[[str], None] | None = None) -> Path:
        """Download the configured model if needed and cache real embeddings."""
        if not self.settings.semantic:
            raise ValueError("Enable semantic search before preparing embeddings.")
        if self.embeddings_ready:
            return self.vectors_path
        if progress:
            progress(f"Encoding {len(self.chunks):,} chunks on CPU. The first run may download a model.")
        model = self._load_encoder()
        vectors = model.encode(
            [chunk.text for chunk in self.chunks],
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.index_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.index_dir / "embeddings.tmp.npy"
        np.save(temporary, vectors)
        temporary.replace(self.vectors_path)
        metadata = {
            "fingerprint": self.fingerprint,
            "model": self.settings.embedding_model,
            "chunks": len(self.chunks),
            "dimensions": int(vectors.shape[1]),
        }
        (self.index_dir / "index.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return self.vectors_path

    def _load_reranker(self):
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "Reranking requires the 'semantic' extra: pip install -e '.[semantic]'"
                ) from exc
            self._reranker = CrossEncoder(
                self.settings.reranker_model,
                device="cpu",
                cache_folder=str(self.project.workspace_path / "models"),
            )
        return self._reranker

    def search(self, question: str, top_k: int | None = None) -> tuple[list[dict], dict]:
        if not question.strip():
            raise ValueError("Enter a question before searching.")
        limit = top_k or self.settings.top_k
        if not 1 <= limit <= 50:
            raise ValueError("Top-k must be between 1 and 50.")
        lexical = self.bm25.get_scores(tokenize(question))
        depth = min(len(self.chunks), max(60, limit * 10))
        eligible = [
            i for i, chunk in enumerate(self.chunks) if set(tokenize(question)) & set(tokenize(chunk.text))
        ]
        lexical_ranking = sorted(eligible, key=lambda i: (-lexical[i], i))[:depth]
        rankings = [lexical_ranking]
        semantic = None
        semantic_ranking: list[int] = []
        if self.settings.semantic:
            if not self.embeddings_ready:
                raise RuntimeError(
                    "Embeddings are not prepared for this data and chunk configuration. "
                    "Open '2 · Build RAG' and select 'Prepare semantic index'."
                )
            vectors = np.load(self.vectors_path, mmap_mode="r")
            query = self._load_encoder().encode(question, normalize_embeddings=True)
            semantic = np.asarray(vectors) @ query
            semantic_ranking = sorted(range(len(self.chunks)), key=lambda i: (-semantic[i], i))[:depth]
            rankings.append(semantic_ranking)

        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, index in enumerate(ranking, 1):
                fused[index] = fused.get(index, 0.0) + 1.0 / (60 + rank)
        ordered = sorted(fused, key=lambda i: (-fused[i], i))
        rerank_scores: dict[int, float] = {}
        if self.settings.rerank:
            candidates = ordered[: min(30, len(ordered))]
            model = self._load_reranker()
            values = model.predict([(question, self.chunks[i].text) for i in candidates])
            rerank_scores = {index: float(score) for index, score in zip(candidates, values)}
            ordered = [i for _, i in sorted(zip(values, candidates), reverse=True)]

        selected_indices = ordered[:limit]
        selected = [self._result_chunk(i, lexical, semantic, fused, rerank_scores) for i in selected_indices]
        for rank, item in enumerate(selected, 1):
            item["selected_rank"] = rank

        candidate_limit = min(len(ordered), max(limit * 3, 10))
        candidates = []
        for rank, index in enumerate(ordered[:candidate_limit], 1):
            candidates.append(
                {
                    "id": self.chunks[index].id,
                    "record_id": self.chunks[index].record_id,
                    "source": self.chunks[index].source,
                    "rank": rank,
                    "selected": index in selected_indices,
                    "rrf_score": float(fused[index]),
                    "keyword_score": float(lexical[index]),
                    "meaning_score": float(semantic[index]) if semantic is not None else None,
                    "reranker_score": rerank_scores.get(index),
                }
            )
        maximum_rrf = sum(1.0 / 61 for _ in rankings)
        for item in selected:
            item["relative_rank_score"] = min(1.0, item["rrf_score"] / maximum_rrf)
        trace = {
            "records_indexed": len(self.records),
            "chunks_indexed": len(self.chunks),
            "chunks_considered": len(fused),
            "search_mode": "BM25 + embeddings + RRF" if semantic is not None else "BM25",
            "reranked": self.settings.rerank,
            "embedding_model": self.settings.embedding_model if semantic is not None else None,
            "chunk_size": self.settings.chunk_size,
            "overlap": self.settings.overlap,
            "top_k": limit,
            "index_fingerprint": self.fingerprint,
            "score_explanation": "RRF is a rank-fusion score, not confidence or correctness.",
            "candidates": candidates,
        }
        return selected, trace

    def _result_chunk(
        self,
        index: int,
        lexical: np.ndarray,
        semantic: np.ndarray | None,
        fused: dict[int, float],
        rerank_scores: dict[int, float],
    ) -> dict:
        value = self.chunks[index].to_dict()
        value.update(
            keyword_score=float(lexical[index]),
            meaning_score=float(semantic[index]) if semantic is not None else None,
            rrf_score=float(fused[index]),
            reranker_score=rerank_scores.get(index),
        )
        return value
