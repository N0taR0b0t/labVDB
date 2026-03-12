from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import NamedTuple

from qdrant_client.models import FilterSelector, PointStruct

from labvdb import (
    CHUNK_PREVIEW_CHARS,
    COLLECTION_NAME,
    MODEL_NAME,
    batched,
    build_filter,
    chunk_page_text,
    compute_doc_id,
    delete_chunks_for_doc,
    elapsed_seconds,
    ensure_collection,
    fetch_doc_ids,  # used by --reconcile only
    get_client,
    load_embedding_model,
    load_manifest_entries,
    make_chunk_id,
    update_manifest_entry,
    write_chunks,
)


class IndexJob(NamedTuple):
    pdf_path: Path
    doc_id: str
    size: int
    mtime_ns: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index PDFs into local Qdrant storage.")
    parser.add_argument("--pdf-dir", type=Path, default=Path("pdfs"))
    parser.add_argument("--file", type=Path, action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--upsert-batch-size", type=int, default=512)
    parser.add_argument("--delete-doc-id")
    parser.add_argument(
        "--reconcile",
        action="store_true",
        help="Compare manifest against Qdrant and report gaps. Read-only repair aid.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def delete_document(client, doc_id: str) -> None:
    doc_filter = build_filter(doc_id=doc_id)
    if doc_filter is None:
        raise ValueError("doc_id is required for deletion")

    # Remove from Qdrant first (removes searchability), then from chunk store.
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=FilterSelector(filter=doc_filter),
        wait=True,
    )
    delete_chunks_for_doc(doc_id)


def resolve_pdf_paths(args: argparse.Namespace) -> list[Path]:
    if args.file:
        return sorted(path for path in args.file if path.suffix.lower() == ".pdf")
    return sorted(args.pdf_dir.glob("*.pdf"))


def plan_index_jobs(
    *,
    client,
    pdf_paths: list[Path],
    force: bool,
) -> tuple[list[IndexJob], dict[str, object]]:
    started_at = time.perf_counter()
    manifest_entries = load_manifest_entries(pdf_paths)

    jobs: list[IndexJob] = []
    skipped_existing: list[tuple[Path, str]] = []
    skipped_duplicate_local: list[tuple[Path, str]] = []
    replaced_doc_ids: list[tuple[Path, str, str]] = []
    planned_doc_ids: set[str] = set()
    metrics = {
        "manifest_hits": 0,
        "hash_computations": 0,
        "hash_seconds": 0.0,
    }

    for pdf_path in pdf_paths:
        resolved_path = str(pdf_path.resolve())
        stat = pdf_path.stat()
        manifest_entry = manifest_entries.get(resolved_path)
        previous_doc_id = manifest_entry.doc_id if manifest_entry and manifest_entry.doc_id else None

        # Resolve doc_id: use manifest cache when size+mtime match, else hash the file.
        doc_id: str | None = None
        if (
            manifest_entry
            and manifest_entry.size == stat.st_size
            and manifest_entry.mtime_ns == stat.st_mtime_ns
            and manifest_entry.doc_id
        ):
            doc_id = manifest_entry.doc_id
            metrics["manifest_hits"] += 1

        if doc_id is None:
            hash_started = time.perf_counter()
            doc_id = compute_doc_id(pdf_path)
            metrics["hash_seconds"] += elapsed_seconds(hash_started)
            metrics["hash_computations"] += 1

        # Dedup within this batch.
        if doc_id in planned_doc_ids:
            skipped_duplicate_local.append((pdf_path, doc_id))
            update_manifest_entry(
                path=pdf_path,
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                doc_id=doc_id,
                indexed_at=None,
                status="duplicate_local",
                last_error=None,
            )
            continue

        # File changed: old doc_id differs from current. Delete the stale vectors.
        # Qdrant delete on a non-existent filter is a safe no-op.
        if previous_doc_id and previous_doc_id != doc_id:
            delete_document(client, previous_doc_id)
            replaced_doc_ids.append((pdf_path, previous_doc_id, doc_id))

        # Manifest says this file is already indexed with the current doc_id — skip unless forced.
        manifest_already_indexed = (
            manifest_entry is not None
            and manifest_entry.status == "indexed"
            and manifest_entry.doc_id == doc_id
            and manifest_entry.size == stat.st_size
            and manifest_entry.mtime_ns == stat.st_mtime_ns
        )
        if manifest_already_indexed and not force:
            skipped_existing.append((pdf_path, doc_id))
            continue

        # --force: delete existing vectors before re-indexing.
        if force and manifest_already_indexed:
            delete_document(client, doc_id)

        jobs.append(
            IndexJob(
                pdf_path=pdf_path,
                doc_id=doc_id,
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
        )
        planned_doc_ids.add(doc_id)

    metrics["plan_seconds"] = elapsed_seconds(started_at)
    return jobs, {
        "skipped_existing": skipped_existing,
        "skipped_duplicate_local": skipped_duplicate_local,
        "replaced_doc_ids": replaced_doc_ids,
        "metrics": metrics,
    }


def flush_records(
    *,
    client,
    model,
    pending_records: list[tuple[int, int, str, str]],
    upsert_batch_size: int,
    metadata: dict[str, object],
    stats: dict[str, float],
    embed_batch_size: int,
) -> None:
    if not pending_records:
        return

    texts = [record[3] for record in pending_records]
    embed_started = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=embed_batch_size,
        normalize_embeddings=True,
    )
    stats["embed_seconds"] += elapsed_seconds(embed_started)

    doc_id = metadata["doc_id"]
    chunk_rows: list[tuple[str, str, str]] = []
    points = []
    for (page_num, chunk_idx, section, text), embedding in zip(pending_records, embeddings):
        chunk_id = make_chunk_id(doc_id, page_num, chunk_idx)
        chunk_rows.append((chunk_id, doc_id, text))
        points.append(
            PointStruct(
                id=chunk_id,
                vector=embedding.tolist(),
                payload={
                    **metadata,
                    "page": page_num,
                    "chunk_idx": chunk_idx,
                    "section": section,
                    "preview": text[:CHUNK_PREVIEW_CHARS],
                },
            )
        )

    # Write full text to SQLite before upserting to Qdrant.
    # If Qdrant upsert fails, text rows are harmless orphans recoverable by re-index.
    # If SQLite write fails, nothing enters Qdrant and the next run retries cleanly.
    write_chunks(chunk_rows)

    upsert_started = time.perf_counter()
    for point_batch in batched(points, upsert_batch_size):
        client.upsert(collection_name=COLLECTION_NAME, points=point_batch, wait=True)
        stats["upsert_batches"] += 1
    stats["upsert_seconds"] += elapsed_seconds(upsert_started)

    stats["chunks_indexed"] += len(points)
    pending_records.clear()


def index_pdf(
    *,
    client,
    model,
    job: IndexJob,
    embed_batch_size: int,
    upsert_batch_size: int,
) -> tuple[dict[str, float], str]:
    import pymupdf as fitz  # deferred: not needed for planning or manifest operations
    doc_started = time.perf_counter()
    doc = fitz.open(job.pdf_path)
    indexed_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "doc_id": job.doc_id,
        "filename": job.pdf_path.name,
        "source_path": str(job.pdf_path.resolve()),
        "indexed_at": indexed_at,
        "total_pages": len(doc),
    }

    stats: dict[str, float] = {
        "chunks_indexed": 0,
        "pages_indexed": 0,
        "extract_seconds": 0.0,
        "embed_seconds": 0.0,
        "upsert_seconds": 0.0,
        "upsert_batches": 0,
    }
    pending_records: list[tuple[int, int, str, str]] = []

    try:
        for page_num in range(len(doc)):
            extract_started = time.perf_counter()
            chunks = chunk_page_text(doc[page_num], page_num + 1)
            stats["extract_seconds"] += elapsed_seconds(extract_started)
            if not chunks:
                continue

            stats["pages_indexed"] += 1
            for chunk in chunks:
                pending_records.append(
                    (chunk.page, chunk.chunk_idx, chunk.section, chunk.text)
                )
                if len(pending_records) >= embed_batch_size:
                    flush_records(
                        client=client,
                        model=model,
                        pending_records=pending_records,
                        upsert_batch_size=upsert_batch_size,
                        metadata=metadata,
                        stats=stats,
                        embed_batch_size=embed_batch_size,
                    )

        flush_records(
            client=client,
            model=model,
            pending_records=pending_records,
            upsert_batch_size=upsert_batch_size,
            metadata=metadata,
            stats=stats,
            embed_batch_size=embed_batch_size,
        )
    finally:
        doc.close()

    stats["doc_seconds"] = elapsed_seconds(doc_started)
    update_manifest_entry(
        path=job.pdf_path,
        size=job.size,
        mtime_ns=job.mtime_ns,
        doc_id=job.doc_id,
        indexed_at=indexed_at,
        status="indexed",
        last_error=None,
    )
    return stats, indexed_at


def print_summary(summary: dict[str, float], metrics: dict[str, float]) -> None:
    total_points = int(summary["total_points"])
    print(
        "Indexing complete: "
        f"{int(summary['indexed'])} indexed, "
        f"{int(summary['skipped_existing'])} already indexed, "
        f"{int(summary['skipped_duplicate_local'])} duplicate local, "
        f"{int(summary['failed'])} failed, "
        f"{total_points} total points"
    )
    print(
        "Metrics: "
        f"plan={metrics['plan_seconds']:.2f}s, "
        f"hash={metrics['hash_seconds']:.2f}s ({int(metrics['hash_computations'])} files), "
        f"embed={summary['embed_seconds']:.2f}s, "
        f"extract={summary['extract_seconds']:.2f}s, "
        f"upsert={summary['upsert_seconds']:.2f}s, "
        f"docs/sec={summary['indexed'] / max(summary['total_seconds'], 1e-9):.2f}, "
        f"chunks/sec={summary['chunks_indexed'] / max(summary['total_seconds'], 1e-9):.2f}"
    )


def reconcile(client) -> None:
    """Compare manifest (indexed status) against Qdrant. Print gaps. Read-only."""
    from labvdb import manifest_stats

    print("Reconcile: scanning Qdrant (this is the only full scroll — maintenance use only)...")
    qdrant_doc_ids = fetch_doc_ids(client)

    with __import__("sqlite3").connect(__import__("labvdb").MANIFEST_PATH) as conn:
        rows = conn.execute(
            "SELECT doc_id FROM manifest WHERE status = 'indexed' AND doc_id IS NOT NULL"
        ).fetchall()
    manifest_doc_ids = {row[0] for row in rows}

    in_qdrant_not_manifest = qdrant_doc_ids - manifest_doc_ids
    in_manifest_not_qdrant = manifest_doc_ids - qdrant_doc_ids

    print(f"Qdrant unique doc_ids:   {len(qdrant_doc_ids)}")
    print(f"Manifest indexed doc_ids: {len(manifest_doc_ids)}")

    if in_qdrant_not_manifest:
        print(f"\nIn Qdrant but NOT in manifest ({len(in_qdrant_not_manifest)}):")
        for doc_id in sorted(in_qdrant_not_manifest):
            print(f"  {doc_id}")
    else:
        print("\nNo doc_ids in Qdrant without a manifest entry.")

    if in_manifest_not_qdrant:
        print(f"\nIn manifest but NOT in Qdrant ({len(in_manifest_not_qdrant)}) — re-index these:")
        for doc_id in sorted(in_manifest_not_qdrant):
            print(f"  {doc_id}")
    else:
        print("No manifest entries missing from Qdrant.")


def main() -> None:
    args = parse_args()
    if args.embed_batch_size <= 0 or args.upsert_batch_size <= 0:
        raise ValueError("Batch sizes must be positive integers")

    overall_started = time.perf_counter()
    client = get_client()
    ensure_collection(client)

    if args.reconcile:
        reconcile(client)
        return

    if args.delete_doc_id:
        delete_document(client, args.delete_doc_id)
        print(f"Deleted document {args.delete_doc_id}")
        return

    pdf_paths = resolve_pdf_paths(args)
    if not pdf_paths:
        raise SystemExit("No PDFs found to index.")

    jobs, planning = plan_index_jobs(client=client, pdf_paths=pdf_paths, force=args.force)
    skipped_existing = planning["skipped_existing"]
    skipped_duplicate_local = planning["skipped_duplicate_local"]
    replaced_doc_ids = planning["replaced_doc_ids"]
    metrics = planning["metrics"]

    if skipped_existing:
        if args.verbose:
            for pdf_path, doc_id in skipped_existing:
                print(f"Skipped {pdf_path.name}: doc_id {doc_id} already indexed")
        else:
            print(f"Skipped {len(skipped_existing)} already indexed files")

    if skipped_duplicate_local:
        if args.verbose:
            for pdf_path, doc_id in skipped_duplicate_local:
                print(f"Skipped {pdf_path.name}: duplicate local doc_id {doc_id}")
        else:
            print(f"Skipped {len(skipped_duplicate_local)} duplicate local files")

    if replaced_doc_ids:
        if args.verbose:
            for pdf_path, previous_doc_id, doc_id in replaced_doc_ids:
                print(
                    f"Replacing {pdf_path.name}: removed stale doc_id {previous_doc_id} "
                    f"before indexing {doc_id}"
                )
        else:
            print(f"Replacing {len(replaced_doc_ids)} changed files with new doc_ids")

    model = None
    if jobs:
        print(f"Loading embedding model: {MODEL_NAME}")
        model = load_embedding_model(MODEL_NAME)

    summary = {
        "indexed": 0,
        "skipped_existing": float(len(skipped_existing)),
        "skipped_duplicate_local": float(len(skipped_duplicate_local)),
        "failed": 0,
        "chunks_indexed": 0.0,
        "extract_seconds": 0.0,
        "embed_seconds": 0.0,
        "upsert_seconds": 0.0,
    }

    for job in jobs:
        try:
            job_stats, _ = index_pdf(
                client=client,
                model=model,
                job=job,
                embed_batch_size=args.embed_batch_size,
                upsert_batch_size=args.upsert_batch_size,
            )
            summary["indexed"] += 1
            summary["chunks_indexed"] += job_stats["chunks_indexed"]
            summary["extract_seconds"] += job_stats["extract_seconds"]
            summary["embed_seconds"] += job_stats["embed_seconds"]
            summary["upsert_seconds"] += job_stats["upsert_seconds"]
            print(
                f"Indexed {job.pdf_path.name}: "
                f"{int(job_stats['chunks_indexed'])} chunks, "
                f"{int(job_stats['pages_indexed'])} pages, "
                f"section-aware chunks in {job_stats['doc_seconds']:.2f}s"
            )
        except Exception as exc:
            summary["failed"] += 1
            update_manifest_entry(
                path=job.pdf_path,
                size=job.size,
                mtime_ns=job.mtime_ns,
                doc_id=job.doc_id,
                indexed_at=None,
                status="failed",
                last_error=str(exc),
            )
            print(f"Failed {job.pdf_path}: {exc}")

    summary["total_points"] = float(
        client.count(collection_name=COLLECTION_NAME, exact=False).count
    )
    summary["total_seconds"] = elapsed_seconds(overall_started)
    print_summary(summary, metrics)


if __name__ == "__main__":
    main()
