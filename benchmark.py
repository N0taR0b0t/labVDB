from __future__ import annotations

import argparse
import os
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import indexer
import labvdb


DEFAULT_QUERIES = [
    "nitro oleic acid cardiovascular disease",
    "acyl-CoA metabolism",
    "sphingolipid plasma neuropathy",
    "stable isotope tracing lipid metabolism",
    "conjugated linoleic acid oxidation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark local indexing and query throughput on a PDF sample."
    )
    parser.add_argument("--pdf-dir", type=Path, default=Path("pdfs"))
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--full-corpus", action="store_true")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--upsert-batch-size", type=int, default=512)
    parser.add_argument("--query-limit", type=int, default=10)
    parser.add_argument("--query-runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument(
        "--model", default=labvdb.MODEL_NAME,
        help="Embedding model to benchmark (default: %(default)s)",
    )
    parser.add_argument(
        "--show-results", type=int, default=0, metavar="N",
        help="After benchmarking, print top N results per query for qualitative review",
    )
    return parser.parse_args()


def configure_paths(temp_root: Path) -> None:
    qdrant_path = temp_root / "qdrant_storage"
    labvdb.QDRANT_PATH = str(qdrant_path)
    labvdb.MANIFEST_PATH = temp_root / "manifest.sqlite3"
    labvdb.CHUNK_STORE_PATH = temp_root / "chunks.sqlite3"
    # Ensure the benchmark always uses embedded Qdrant, never the production server.
    os.environ.pop("QDRANT_URL", None)


def sample_pdfs(pdf_dir: Path, sample_size: int) -> list[Path]:
    pdfs = sorted(path for path in pdf_dir.glob("*.pdf") if path.is_file())
    if not pdfs:
        raise SystemExit(f"No PDFs found in {pdf_dir}")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    return pdfs[: min(sample_size, len(pdfs))]


def benchmark_indexing(
    *,
    client,
    model,
    pdf_paths: list[Path],
    embed_batch_size: int,
    upsert_batch_size: int,
) -> dict[str, float]:
    started_at = time.perf_counter()
    jobs, planning = indexer.plan_index_jobs(client=client, pdf_paths=pdf_paths, force=False)

    summary = {
        "docs_planned": float(len(jobs)),
        "docs_indexed": 0.0,
        "chunks_indexed": 0.0,
        "pages_indexed": 0.0,
        "extract_seconds": 0.0,
        "embed_seconds": 0.0,
        "upsert_seconds": 0.0,
        "plan_seconds": float(planning["metrics"]["plan_seconds"]),
        "hash_seconds": float(planning["metrics"]["hash_seconds"]),
    }

    for job in jobs:
        job_stats, _ = indexer.index_pdf(
            client=client,
            model=model,
            job=job,
            embed_batch_size=embed_batch_size,
            upsert_batch_size=upsert_batch_size,
        )
        summary["docs_indexed"] += 1
        summary["chunks_indexed"] += float(job_stats["chunks_indexed"])
        summary["pages_indexed"] += float(job_stats["pages_indexed"])
        summary["extract_seconds"] += float(job_stats["extract_seconds"])
        summary["embed_seconds"] += float(job_stats["embed_seconds"])
        summary["upsert_seconds"] += float(job_stats["upsert_seconds"])

    summary["total_seconds"] = time.perf_counter() - started_at
    summary["point_count"] = float(
        client.count(collection_name=labvdb.COLLECTION_NAME, exact=False).count
    )
    return summary


def run_query_once(
    *,
    client,
    model,
    query: str,
    limit: int,
) -> dict[str, float]:
    encode_started = time.perf_counter()
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    encode_seconds = time.perf_counter() - encode_started

    dense_started = time.perf_counter()
    dense_results = client.query_points(
        collection_name=labvdb.COLLECTION_NAME,
        query=query_vector,
        limit=limit,
    ).points
    dense_seconds = time.perf_counter() - dense_started

    hybrid_started = time.perf_counter()
    hybrid_results = labvdb.rerank_hybrid(query, dense_results, limit)
    hybrid_seconds = time.perf_counter() - hybrid_started

    return {
        "encode_seconds": encode_seconds,
        "dense_seconds": dense_seconds,
        "hybrid_seconds": hybrid_seconds,
        "result_count": float(len(dense_results)),
        "hybrid_result_count": float(len(hybrid_results)),
    }


def benchmark_queries(
    *,
    client,
    model,
    queries: list[str],
    limit: int,
    query_runs: int,
    warmup_runs: int,
) -> dict[str, float]:
    for _ in range(warmup_runs):
        for query in queries:
            run_query_once(client=client, model=model, query=query, limit=limit)

    encode_samples: list[float] = []
    dense_samples: list[float] = []
    hybrid_samples: list[float] = []
    total_samples: list[float] = []

    for _ in range(query_runs):
        for query in queries:
            started_at = time.perf_counter()
            result = run_query_once(client=client, model=model, query=query, limit=limit)
            total_samples.append(time.perf_counter() - started_at)
            encode_samples.append(result["encode_seconds"])
            dense_samples.append(result["dense_seconds"])
            hybrid_samples.append(result["hybrid_seconds"])

    sample_count = len(total_samples)
    total_query_seconds = sum(total_samples)

    return {
        "query_samples": float(sample_count),
        "encode_p50_ms": statistics.median(encode_samples) * 1000.0,
        "dense_p50_ms": statistics.median(dense_samples) * 1000.0,
        "hybrid_p50_ms": statistics.median(hybrid_samples) * 1000.0,
        "total_p50_ms": statistics.median(total_samples) * 1000.0,
        "encode_mean_ms": statistics.fmean(encode_samples) * 1000.0,
        "dense_mean_ms": statistics.fmean(dense_samples) * 1000.0,
        "hybrid_mean_ms": statistics.fmean(hybrid_samples) * 1000.0,
        "total_mean_ms": statistics.fmean(total_samples) * 1000.0,
        "queries_per_second": sample_count / max(total_query_seconds, 1e-9),
    }


def show_query_results(*, client, model, queries: list[str], limit: int) -> None:
    for query in queries:
        print(f"\n  query: {query!r}")
        vec = model.encode(query, normalize_embeddings=True).tolist()
        dense = client.query_points(
            collection_name=labvdb.COLLECTION_NAME,
            query=vec,
            limit=limit * 3,
        ).points
        results = labvdb.rerank_hybrid(query, dense, limit)
        for i, r in enumerate(results, 1):
            fname = r["filename"][:55] + "…" if len(r["filename"]) > 55 else r["filename"]
            print(f"  {i:2}. [{r['score']:.3f}]  {fname}  p{r['page']}  §{r['section']}")
            preview = r["preview"].replace("\n", " ")
            print(f"       {preview[:120]}…")


def format_seconds(value: float) -> str:
    return f"{value:.2f}s"


def format_ms(value: float) -> str:
    return f"{value:.1f} ms"


def main() -> None:
    args = parse_args()
    pdf_paths = sample_pdfs(
        args.pdf_dir,
        len(list(args.pdf_dir.glob("*.pdf"))) if args.full_corpus else args.sample_size,
    )
    queries = args.query or list(DEFAULT_QUERIES)

    # Override model globals before collection creation so VECTOR_SIZE is correct.
    labvdb.MODEL_NAME = args.model

    temp_root = Path(tempfile.mkdtemp(prefix="labvdb-bench-"))
    client = None

    try:
        configure_paths(temp_root)

        print(f"Model: {args.model}")
        print(f"Loading model…")
        model = labvdb.load_embedding_model(args.model)
        labvdb.VECTOR_SIZE = int(model.encode("x", normalize_embeddings=True).shape[0])
        print(f"Vector size: {labvdb.VECTOR_SIZE}")

        client = labvdb.get_client()
        labvdb.ensure_collection(client)

        mode = "full corpus" if args.full_corpus else "sample"
        print(f"Benchmark mode: {mode}")
        print(f"Benchmark sample: {len(pdf_paths)} PDFs")
        print(f"Temporary benchmark store: {temp_root}")

        indexing = benchmark_indexing(
            client=client,
            model=model,
            pdf_paths=pdf_paths,
            embed_batch_size=args.embed_batch_size,
            upsert_batch_size=args.upsert_batch_size,
        )
        queries_summary = benchmark_queries(
            client=client,
            model=model,
            queries=queries,
            limit=args.query_limit,
            query_runs=args.query_runs,
            warmup_runs=args.warmup_runs,
        )

        print("\nIndexing")
        print(f"  docs indexed: {int(indexing['docs_indexed'])}/{int(indexing['docs_planned'])}")
        print(f"  chunks indexed: {int(indexing['chunks_indexed'])}")
        print(f"  pages indexed: {int(indexing['pages_indexed'])}")
        print(f"  total points: {int(indexing['point_count'])}")
        print(f"  total time: {format_seconds(indexing['total_seconds'])}")
        print(f"  planning time: {format_seconds(indexing['plan_seconds'])}")
        print(f"  hash time: {format_seconds(indexing['hash_seconds'])}")
        print(f"  extract time: {format_seconds(indexing['extract_seconds'])}")
        print(f"  embed time: {format_seconds(indexing['embed_seconds'])}")
        print(f"  upsert time: {format_seconds(indexing['upsert_seconds'])}")
        print(
            "  throughput: "
            f"{indexing['docs_indexed'] / max(indexing['total_seconds'], 1e-9):.2f} docs/s, "
            f"{indexing['chunks_indexed'] / max(indexing['total_seconds'], 1e-9):.2f} chunks/s"
        )
        print(
            "  density: "
            f"{indexing['chunks_indexed'] / max(indexing['docs_indexed'], 1e-9):.1f} chunks/doc, "
            f"{indexing['pages_indexed'] / max(indexing['docs_indexed'], 1e-9):.1f} pages/doc"
        )

        print("\nQueries")
        print(f"  queries used: {len(queries)}")
        print(f"  measured runs: {int(queries_summary['query_samples'])}")
        print(f"  encode p50: {format_ms(queries_summary['encode_p50_ms'])}")
        print(f"  dense search p50: {format_ms(queries_summary['dense_p50_ms'])}")
        print(f"  hybrid rerank p50: {format_ms(queries_summary['hybrid_p50_ms'])}")
        print(f"  total p50: {format_ms(queries_summary['total_p50_ms'])}")
        print(f"  total mean: {format_ms(queries_summary['total_mean_ms'])}")
        print(f"  throughput: {queries_summary['queries_per_second']:.2f} queries/s")

        if args.show_results > 0:
            print(f"\nTop {args.show_results} results per query (hybrid reranked):")
            show_query_results(client=client, model=model, queries=queries, limit=args.show_results)

    finally:
        if client is not None:
            client.close()
        if not args.keep_artifacts:
            shutil.rmtree(temp_root, ignore_errors=True)
        else:
            print(f"\nKept benchmark artifacts at {temp_root}")


if __name__ == "__main__":
    main()
