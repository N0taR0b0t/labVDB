from __future__ import annotations

import functools
import hashlib
import math
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    VectorParams,
)

COLLECTION_NAME = "pdfs"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QDRANT_PATH = "qdrant_storage"
VECTOR_SIZE = 384
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
MANIFEST_PATH = Path("manifest.sqlite3")
CHUNK_TARGET_CHARS = 1800
CHUNK_MIN_CHARS = 500
CHUNK_OVERLAP_BLOCKS = 1
SECTION_ALIASES = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "background": "Introduction",
    "methods": "Methods",
    "materials and methods": "Methods",
    "methodology": "Methods",
    "results": "Results",
    "discussion": "Discussion",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "references": "References",
}
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    size: int
    mtime_ns: int
    doc_id: str | None
    indexed_at: str | None
    status: str
    last_error: str | None


@dataclass(frozen=True)
class ChunkRecord:
    page: int
    chunk_idx: int
    section: str
    text: str


def get_client() -> QdrantClient:
    return QdrantClient(path=QDRANT_PATH)


def ensure_collection(client: QdrantClient) -> None:
    try:
        client.get_collection(COLLECTION_NAME)
    except Exception:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    ensure_payload_indexes(client)


def ensure_payload_indexes(client: QdrantClient) -> None:
    index_specs = {
        "doc_id": PayloadSchemaType.KEYWORD,
        "filename": PayloadSchemaType.KEYWORD,
        "section": PayloadSchemaType.KEYWORD,
        "page": PayloadSchemaType.INTEGER,
        "indexed_at": PayloadSchemaType.DATETIME,
    }
    for field_name, schema in index_specs.items():
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            )
        except Exception:
            continue


def model_cache_path(model_name: str = MODEL_NAME) -> Path:
    repo_dir = model_name.replace("/", "--")
    return HF_CACHE_DIR / f"models--{repo_dir}"


def model_is_cached(model_name: str = MODEL_NAME) -> bool:
    cache_dir = model_cache_path(model_name)
    refs_main = cache_dir / "refs" / "main"
    snapshots_dir = cache_dir / "snapshots"
    if not refs_main.exists() or not snapshots_dir.exists():
        return False

    revision = refs_main.read_text().strip()
    return (snapshots_dir / revision).is_dir()


def cached_model_snapshot_path(model_name: str = MODEL_NAME) -> Path | None:
    cache_dir = model_cache_path(model_name)
    refs_main = cache_dir / "refs" / "main"
    snapshots_dir = cache_dir / "snapshots"
    if not refs_main.exists() or not snapshots_dir.exists():
        return None

    revision = refs_main.read_text().strip()
    snapshot_path = snapshots_dir / revision
    if not snapshot_path.is_dir():
        return None
    return snapshot_path


@functools.lru_cache(maxsize=1)
def load_embedding_model(model_name: str = MODEL_NAME):
    from sentence_transformers import SentenceTransformer

    cached_snapshot = cached_model_snapshot_path(model_name)
    if cached_snapshot is not None:
        return SentenceTransformer(str(cached_snapshot))

    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load embedding model '{model_name}'. "
            "No cached local snapshot was found, and online resolution failed. "
            "Cache the model locally or enable network access before retrying."
        ) from exc


def ensure_manifest_db() -> None:
    with sqlite3.connect(MANIFEST_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS manifest (
                path TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                doc_id TEXT,
                indexed_at TEXT,
                status TEXT NOT NULL,
                last_error TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_manifest_doc_id ON manifest(doc_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_manifest_status ON manifest(status)"
        )
        conn.commit()


def load_manifest_entries(paths: Sequence[Path]) -> dict[str, ManifestEntry]:
    ensure_manifest_db()
    path_strings = [str(path.resolve()) for path in paths]
    if not path_strings:
        return {}

    placeholders = ",".join("?" for _ in path_strings)
    query = (
        "SELECT path, size, mtime_ns, doc_id, indexed_at, status, last_error "
        f"FROM manifest WHERE path IN ({placeholders})"
    )
    with sqlite3.connect(MANIFEST_PATH) as conn:
        rows = conn.execute(query, path_strings).fetchall()

    return {
        row[0]: ManifestEntry(
            path=row[0],
            size=row[1],
            mtime_ns=row[2],
            doc_id=row[3],
            indexed_at=row[4],
            status=row[5],
            last_error=row[6],
        )
        for row in rows
    }


def update_manifest_entry(
    *,
    path: Path,
    size: int,
    mtime_ns: int,
    doc_id: str | None,
    indexed_at: str | None,
    status: str,
    last_error: str | None,
) -> None:
    ensure_manifest_db()
    with sqlite3.connect(MANIFEST_PATH) as conn:
        conn.execute(
            """
            INSERT INTO manifest(path, size, mtime_ns, doc_id, indexed_at, status, last_error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                size=excluded.size,
                mtime_ns=excluded.mtime_ns,
                doc_id=excluded.doc_id,
                indexed_at=excluded.indexed_at,
                status=excluded.status,
                last_error=excluded.last_error
            """,
            (
                str(path.resolve()),
                size,
                mtime_ns,
                doc_id,
                indexed_at,
                status,
                last_error,
            ),
        )
        conn.commit()


def manifest_stats() -> dict[str, int]:
    ensure_manifest_db()
    with sqlite3.connect(MANIFEST_PATH) as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM manifest GROUP BY status"
        ).fetchall()
    return {status: count for status, count in rows}


def compute_doc_id(pdf_path: Path) -> str:
    digest = hashlib.sha1()
    with pdf_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_chunk_id(doc_id: str, page_num: int, chunk_idx: int) -> str:
    key = f"{doc_id}:{page_num}:{chunk_idx}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def build_filter(
    *,
    doc_id: str | None = None,
    filename: str | None = None,
    section: str | None = None,
) -> Filter | None:
    conditions = []

    if doc_id:
        conditions.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
    if filename:
        conditions.append(
            FieldCondition(key="filename", match=MatchValue(value=filename))
        )
    if section:
        conditions.append(FieldCondition(key="section", match=MatchValue(value=section)))

    if not conditions:
        return None

    return Filter(must=conditions)


def batched(items: Iterable, batch_size: int) -> Iterator[list]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def count_unique_doc_ids(client: QdrantClient, page_size: int = 256) -> int:
    return len(fetch_doc_ids(client, page_size=page_size))


def fetch_doc_ids(client: QdrantClient, page_size: int = 256) -> set[str]:
    doc_ids: set[str] = set()
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=["doc_id"],
            with_vectors=False,
            limit=page_size,
            offset=offset,
        )

        for point in points:
            doc_id = point.payload.get("doc_id")
            if doc_id:
                doc_ids.add(doc_id)

        if offset is None:
            break

    return doc_ids


def canonicalize_section(text: str) -> str | None:
    candidate = re.sub(r"[^A-Za-z ]+", " ", text).strip().lower()
    candidate = re.sub(r"\s+", " ", candidate)
    return SECTION_ALIASES.get(candidate)


def clean_block_text(text: str) -> str:
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def page_blocks(page) -> list[str]:
    blocks = page.get_text("blocks")
    ordered = sorted(blocks, key=lambda block: (round(block[1], 1), round(block[0], 1)))
    texts = []
    for block in ordered:
        text = clean_block_text(block[4])
        if text:
            texts.append(text)
    return texts


def chunk_page_text(page, page_num: int, default_section: str = "Unknown") -> list[ChunkRecord]:
    blocks = page_blocks(page)
    if not blocks:
        return []

    section = default_section
    paragraphs: list[tuple[str, str]] = []
    for block in blocks:
        heading = canonicalize_section(block)
        if heading and len(block.split()) <= 8:
            section = heading
            continue
        paragraphs.append((section, block))

    chunks: list[ChunkRecord] = []
    chunk_idx = 0
    cursor = 0
    while cursor < len(paragraphs):
        overlap_start = max(0, cursor - CHUNK_OVERLAP_BLOCKS)
        current_section = paragraphs[cursor][0]
        collected: list[str] = []
        total_chars = 0
        for overlap_idx in range(overlap_start, cursor):
            para_section, para_text = paragraphs[overlap_idx]
            if para_section != current_section:
                continue
            collected.append(para_text)
            total_chars += len(para_text)
        end = cursor

        while end < len(paragraphs):
            para_section, para_text = paragraphs[end]
            if collected and para_section != current_section and total_chars >= CHUNK_MIN_CHARS:
                break
            if total_chars and total_chars + len(para_text) > CHUNK_TARGET_CHARS:
                break

            collected.append(para_text)
            total_chars += len(para_text)
            end += 1

        if not collected:
            collected.append(paragraphs[cursor][1])
            end = cursor + 1

        text = "\n\n".join(collected).strip()
        if text:
            chunks.append(
                ChunkRecord(
                    page=page_num,
                    chunk_idx=chunk_idx,
                    section=current_section,
                    text=text,
                )
            )
            chunk_idx += 1

        cursor = max(cursor + 1, end - CHUNK_OVERLAP_BLOCKS)

    return chunks


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def lexical_score(query: str, text: str) -> float:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    text_tokens = tokenize(text)
    if not text_tokens:
        return 0.0

    text_token_set = set(text_tokens)
    matched = sum(1 for token in set(query_tokens) if token in text_token_set)
    coverage = matched / len(set(query_tokens))
    frequency = sum(text_tokens.count(token) for token in query_tokens) / len(query_tokens)
    phrase_bonus = 0.2 if query.lower() in text.lower() else 0.0
    return min(1.0, coverage * 0.7 + min(frequency / 3.0, 0.3) + phrase_bonus)


def rerank_hybrid(query: str, results: list, limit: int) -> list[dict[str, object]]:
    if not results:
        return []

    dense_scores = [float(result.score) for result in results]
    min_dense = min(dense_scores)
    max_dense = max(dense_scores)
    spread = max(max_dense - min_dense, 1e-9)

    reranked = []
    for result in results:
        text = result.payload["text"]
        dense = (float(result.score) - min_dense) / spread if spread > 0 else 1.0
        lexical = lexical_score(query, text)
        final_score = dense * 0.75 + lexical * 0.25
        reranked.append(
            {
                "score": final_score,
                "dense_score": float(result.score),
                "lexical_score": lexical,
                "doc_id": result.payload["doc_id"],
                "filename": result.payload["filename"],
                "page": result.payload["page"],
                "chunk_idx": result.payload["chunk_idx"],
                "section": result.payload.get("section", "Unknown"),
                "text": text,
            }
        )

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:limit]


def elapsed_seconds(start_time: float) -> float:
    return time.perf_counter() - start_time
