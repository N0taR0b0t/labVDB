# labVDB

Semantic search over a lab PDF corpus. Drop papers into a folder, run the indexer, search via a browser or `curl`. Designed to run on a cheap CPU-only VPS or a lab workstation under a $30/month compute budget.

## What it does

- Extracts text from PDFs, chunks it by section, and embeds each chunk with `BAAI/bge-small-en-v1.5`
- Stores vectors in Qdrant (server mode) and full chunk text in SQLite
- Serves a search API with hybrid dense + lexical reranking and a minimal browser UI
- Skips unchanged files on re-runs; SHA1 content hashes detect duplicates regardless of filename

## Architecture

| Component | Role |
|-----------|------|
| `Qdrant` (docker-compose) | Vector search; lightweight payloads (200-char preview only) |
| `manifest.sqlite3` | Source of truth for document inventory and indexing state |
| `chunks.sqlite3` | Full chunk text, fetched on demand via `/chunk/{chunk_id}` |
| `search_api.py` | FastAPI app — search, stats, chunk fetch, browser UI |
| `indexer.py` | CLI — index PDFs, fix filenames, delete documents, reconcile |
| `labvdb.py` | Core library shared by both |

## Prerequisites

- Docker and docker-compose
- Python 3.12
- `python3-pip` and `python3.12-venv` (Ubuntu: `sudo apt install python3-pip python3.12-venv`)

## Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd labVDB

# 2. Create virtualenv and install dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# .env contains: QDRANT_URL=http://localhost:6333

# 4. Start Qdrant
docker-compose up -d

# 5. Start the API server
set -a && source .env && set +a
nohup .venv/bin/uvicorn search_api:app --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &
```

> **Important:** `QDRANT_URL` must be in the environment when starting uvicorn or the indexer. Always `source .env` first, or the app silently falls back to embedded Qdrant.

## Indexing PDFs

Place PDFs in the `pdfs/` directory (create it if needed), then run the indexer:

```bash
set -a && source .env && set +a

# First run (or after extracting a ZIP from Windows)
PYTHONUNBUFFERED=1 .venv/bin/python indexer.py --fix-filenames --pdf-dir pdfs/

# Normal incremental index (skips already-indexed files)
PYTHONUNBUFFERED=1 .venv/bin/python indexer.py --pdf-dir pdfs/
```

Use `PYTHONUNBUFFERED=1` so progress lines appear in real time rather than buffering until completion.

### Adding new PDFs

Drop new files into `pdfs/` and re-run the indexer. Only new or changed files are processed; everything else is skipped in seconds.

```bash
cp /path/to/new-paper.pdf pdfs/
PYTHONUNBUFFERED=1 .venv/bin/python indexer.py --pdf-dir pdfs/
```

### Fixing filenames from Windows ZIPs

PDFs extracted from Windows ZIPs on Linux often have garbled filenames (CP437/UTF-8 mojibake). Fix them before indexing:

```bash
.venv/bin/python indexer.py --fix-filenames --pdf-dir pdfs/
```

This renames files in place and then exits. Run it once after any new ZIP extraction.

## Searching

### Browser UI

Open `http://<host>:8000` in a browser. Type a query and optionally filter by filename, doc ID, or section. Click any result to expand the full chunk text.

### API

```bash
# Basic search
curl "http://localhost:8000/search?q=fatty+acid+oxidation"

# Limit results
curl "http://localhost:8000/search?q=sphingolipid+metabolism&limit=5"

# Dense-only (disable hybrid reranking)
curl "http://localhost:8000/search?q=acyl-CoA&hybrid=false"

# Filter by section
curl "http://localhost:8000/search?q=statistical+analysis&section=Methods"

# Filter by filename (exact match)
curl "http://localhost:8000/search?q=lipid+droplet&filename=Smith2021.pdf"

# Filter by doc_id (SHA1 hash — disables per-doc deduplication cap)
curl "http://localhost:8000/search?q=ketogenesis&doc_id=<sha1>"

# Control max results per document (default 2)
curl "http://localhost:8000/search?q=beta+oxidation&max_per_doc=5"

# Fetch full text for a chunk
curl "http://localhost:8000/chunk/<chunk_id>"

# Stats
curl "http://localhost:8000/stats"

# Health check
curl "http://localhost:8000/health"
```

### Search response fields

| Field | Description |
|-------|-------------|
| `score` | Hybrid score (dense×0.75 + lexical×0.25) |
| `dense_score` | Raw cosine similarity from Qdrant |
| `lexical_score` | Token overlap + phrase bonus against the preview |
| `filename` | PDF filename (sanitized) |
| `doc_id` | SHA1 content hash of the source PDF |
| `page` | 1-indexed page number |
| `section` | Detected section (Abstract, Methods, Results, …) |
| `chunk_idx` | Chunk index within the page |
| `chunk_id` | UUID — use with `/chunk/{chunk_id}` to get full text |
| `preview` | First 200 characters of the chunk |

## Indexer CLI reference

```
python indexer.py [options]

--pdf-dir PATH          Directory of PDFs to index (default: pdfs/)
--file PATH             Index a specific file (repeatable; overrides --pdf-dir)
--force                 Re-index all files, even if already indexed
--fix-filenames         Fix CP437/UTF-8 mojibake in filenames, then exit
--delete-doc-id HASH    Remove a document by SHA1 hash from Qdrant and chunk store
--reconcile             Compare manifest against Qdrant and report gaps (read-only)
--embed-batch-size N    Embedding batch size (default: 32)
--upsert-batch-size N   Qdrant upsert batch size (default: 512)
--verbose               Print one line per skipped file instead of a summary count
```

### Deleting a document

```bash
.venv/bin/python indexer.py --delete-doc-id <sha1-hash>
```

The doc_id appears in every search result and in `manifest.sqlite3`.

### Reconciling manifest vs Qdrant

```bash
set -a && source .env && set +a
.venv/bin/python indexer.py --reconcile
```

Prints doc_ids that are in one store but not the other. Read-only — does not modify anything. Use this after a partial failure or manual intervention to identify what needs re-indexing.

## Tests

```bash
.venv/bin/python -m pytest tests/test_labvdb_core.py -v
```

Covers: doc_id stability and content-based deduplication, `is_junk_block` (header/footer suppression), and `canonicalize_section` (section detection with numbered headings and multi-word variants).

## Benchmark

```bash
set -a && source .env && set +a

# Quick benchmark on 12 PDFs from pdfs/
.venv/bin/python benchmark.py

# Full corpus benchmark
.venv/bin/python benchmark.py --full-corpus

# Custom queries
.venv/bin/python benchmark.py --query "nitro oleic acid" --query "lipid droplet formation"
```

The benchmark runs in a temporary isolated store — it does not touch the production index.

## Known limitations

- **No authentication.** The API is open. Suitable for single-user or trusted-network deployments only.
- **No GPU required.** Indexing and query encoding run on CPU. Indexing throughput is modest (~1–3 docs/sec depending on hardware and PDF size).
- **Section detection is heuristic.** Section labels are detected from heading text using a curated alias table and sliding-window prefix matching. Unusual heading formats will fall through as `Unknown`.
- **Embedded Qdrant fallback.** If `QDRANT_URL` is not set, the app falls back to an embedded local Qdrant instance in `qdrant_storage/`. This is fine for development but not recommended for production use with large corpora.
- **Full re-index required when switching from embedded to server Qdrant.** The manifest tracks all state, so `indexer.py --force` after deploying server Qdrant is the migration path.
