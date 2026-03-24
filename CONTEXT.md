# Project Context

## Primary Goal

Build a lab-specific PDF search tool for scientific papers that supports:

- rapid local indexing
- semantic passage search
- metadata-aware filtering
- very low operating cost

This is a research workflow tool, not a general-purpose search product.

## Hard Constraint

The system must be viable under a **$30/month compute budget**.

That budget constraint is now a first-class design requirement, not an optimization target.

## What The Budget Constraint Means

The design must assume:

- no always-on GPU rental
- no high-performance dedicated inference server
- no distributed infrastructure
- no architecture that depends on expensive reindexing

The tool should be able to run using:

- a cheap CPU-only VPS, or
- an existing lab workstation / desktop, or
- a local machine with occasional batch indexing

## Corpus Expectations

- Current corpus: 182 PDFs
- Future target: roughly 15,000 PDFs
- Domain: scientific / biomedical / metabolism / lipid / assay / lab literature

## Product Requirements

### Retrieval

- Search should work semantically.
- Search should still preserve exact-match usefulness for scientific terms, abbreviations, lipids, assay names, gene names, and units.
- Results should return useful passages or chunks, not just document names.

### Indexing

- Re-runs must be cheap.
- The system should avoid reprocessing unchanged files.
- Duplicate files should not be indexed repeatedly.
- Full reindexing should be rare and treated as a batch operation.

### Operations

- The system must stay simple enough for rapid prototyping by a small lab team.
- It should remain understandable and debuggable without DevOps-heavy infrastructure.
- The default operating mode should be CPU-friendly.

## Architectural Direction

The current architecture after Phases 1–3:

1. Qdrant server (via `QDRANT_URL`) for vector search only — lightweight payloads, no full text.
2. `manifest.sqlite3` as authoritative source of truth for document inventory and indexing state.
3. `chunks.sqlite3` for full chunk text, keyed by `chunk_id` — fetched on demand, not during search.
4. Content-based `doc_id` (SHA1) for deduplication and reindex safety.
5. Payload indexes on `doc_id`, `filename`, `section`, `page`, `indexed_at` — active in server mode.
6. Filename sanitization at ingest: `sanitize_filename()` in `labvdb.py` corrects CP437/UTF-8 ZIP extraction mojibake before filenames are stored in Qdrant payloads. The filesystem path used for change detection is unaffected. `indexer.py --fix-filenames` renames files on disk to match (22 files corrected on first production run).
7. Hybrid retrieval active: `/search` calls `rerank_hybrid` by default (dense×0.75 + lexical×0.25); `?hybrid=false` for dense-only comparison.
8. Section filter active: `/search` accepts `?section=`; UI has a section input field. Section metadata is now populated during indexing (re-index in progress as of 2026-03-19).
9. Manifest stats in `/stats`: returns per-status counts from manifest.sqlite3.
10. Header/footer suppression: `is_junk_block()` drops low-information blocks at ingest (running headers, bare page numbers, <3 distinct tokens).
11. Per-document deduplication in `/search`: `?max_per_doc=` (default 2) caps chunks per document after reranking; cap skipped when `doc_id`/`filename` filter is active.
12. Click-to-expand in UI: result previews expand to full chunk text on click via `/chunk/{chunk_id}`.
13. Tests: `tests/test_labvdb_core.py` covers doc_id stability, duplicate detection, `is_junk_block`, and `canonicalize_section` (22 tests, all passing).

## Budget-Adjusted Architecture Principles

### What to prefer

- CPU-first serving
- local disk-backed vector database
- small or medium embedding models
- incremental indexing
- offline or batch-heavy workflows
- hybrid retrieval techniques that do not require large always-on models

### What to avoid

- always-on rented GPU instances
- large-model dependencies for core operation
- expensive reranking pipelines as a mandatory runtime step
- infrastructure that assumes cloud-scale throughput

## Model Direction Under A $30 Cap

### Current practical choice

`BAAI/bge-small-en-v1.5` is still a reasonable prototype model under this budget because:

- it is relatively lightweight
- it can run on CPU
- it supports local experimentation

### What this implies

- indexing will be slower than GPU-backed setups
- quality must be improved with chunking, metadata, filters, and hybrid ranking, not just by paying for a larger model
- model selection must balance retrieval quality against CPU cost

### Model recommendation under current constraints

- Keep a small dense model for now.
- Prefer improving retrieval quality through:
  - better chunking
  - metadata filters
  - lexical scoring
  - lightweight hybrid ranking
- Do not switch to `bge-m3`. Evaluated 2026-03-24 on 4-core EPYC VPS: embed phase took >10 min for 12 docs (vs ~190s for bge-small), making it 10–20× slower. At 15k PDFs indexing time would be measured in weeks. Rejected.
- Do not design around large always-on scientific embedding models that assume bigger hardware.

## Retrieval Strategy Under Budget Constraints

The system should not assume “bigger model = better solution.”

Under a strict cost cap, the strongest path is usually:

1. chunk documents well
2. preserve section metadata
3. add metadata filters
4. combine dense similarity with lexical matching
5. optionally add reranking later, but not as a mandatory always-on expensive step

## Operational Lessons Already Learned

- Dense-only retrieval is not enough for a scientific corpus.
- Cheap reruns matter more than raw one-time indexing speed.
- Content-based document identity is required.
- Model loading should be lazy and cache-aware.
- Index planning should avoid unnecessary repeated database checks.
- PDFs unzipped from Windows ZIPs on Linux frequently have CP437/UTF-8 mojibake in filenames. Run `--fix-filenames` after any new ZIP extraction before indexing.
- On Ubuntu 24.04, pip is not available by default — install python3-pip and python3.12-venv, then use a venv.
- `QDRANT_URL` is read from the environment at call time, not from `.env` automatically. Always `source .env` before starting uvicorn or the indexer, or the app silently falls back to embedded Qdrant.
- Run the indexer with `PYTHONUNBUFFERED=1` so per-document log lines are written in real time rather than buffered until completion.
- Section detection required both expanding `SECTION_ALIASES` and threading `current_section` across pages in `index_pdf`. Without cross-page carry, every page after the first reset to "Unknown" even when the section hadn't changed.
- Running page headers (e.g. "FATTY ACID OXIDATION AND KETOGENESIS\n403") are a recurring chunk quality issue in scientific PDFs. They pass normal length checks but are caught by the two-line uppercase+page-number heuristic in `is_junk_block`.
- `canonicalize_section` needs both a larger alias table and a sliding-window prefix match to handle numbered headings ("3.2 Statistical Analysis") and multi-word variants ("Patients and Methods"). Exact-match alone catches only a small fraction of real biomedical section headings.

## Planned Architecture Pivot for Scale

The system is being redesigned for reliable operation at ~20,000 PDFs on a single low-cost machine.

### Core design change

Stop treating Qdrant as both the search engine and the system-of-record.

| Component | Role |
|-----------|------|
| Qdrant server | vector search only |
| SQLite manifest DB | document inventory, indexing state, file change tracking, stats |
| SQLite chunk store | full chunk text storage |

### Implementation phases

#### Critical (must complete before scaling) — all three complete

**Phase 1 — Deploy server Qdrant with disk-backed operation** ✓

`get_client()` reads `QDRANT_URL` from env at call time; connects to server if set, embedded fallback if absent. New collections created with scalar quantization (INT8) and `on_disk_payload=True`. `ensure_payload_indexes()` catches only 400 responses (index already exists); real errors propagate. FastAPI client moved into lifespan handler. `/health` endpoint added. `docker-compose.yml` and `.env.example` provided.

**Phase 2 — Make the manifest authoritative; eliminate whole-collection scans** ✓

`plan_index_jobs()` uses manifest `(status, doc_id, size, mtime_ns)` for all skip/change decisions — no `fetch_doc_ids()` scroll. Changed-file delete fires unconditionally (Qdrant no-op if point absent). `count_indexed_docs()` (SQLite `COUNT`) replaces scroll-based document counting in `/stats`. `--reconcile` CLI flag performs the only permitted full scroll, explicitly as a maintenance tool.

**Phase 3 — Remove full chunk text from Qdrant payloads** ✓

Full chunk text stored in `chunks.sqlite3` (table: `chunk_id, doc_id, full_text`). SQLite write happens before Qdrant upsert in `flush_records()`. Qdrant payload carries `preview` (200 chars) only — no `text` field. `delete_document()` cleans Qdrant then chunk store. `/search` returns `preview` + `chunk_id`. New `GET /chunk/{chunk_id}` endpoint fetches full text from SQLite on demand; returns 404 if not found.

#### Supplemental (improves perceived speed; not a prerequisite for scaling)

**Phase 4 — Progressive result loading and UI filters**

Return only `filename`, `page`, `section`, short snippet, and score in the initial search response. Load full chunk text only when the user opens a result. Add fast filters in the UI for filename, document, and section.

### Migration note

Switching from embedded to server Qdrant requires a full re-index pass. The manifest tracks all files and their statuses, so `indexer.py --force` after deploying the new code is the migration path. No data migration from the embedded store is needed.

## Success Criteria

This project is on the right track if it becomes:

- usable for a small lab under a $30/month compute budget
- cheap to rerun after folder updates
- accurate enough to retrieve scientifically relevant passages
- simple enough to run locally or on a cheap CPU machine
- scalable in data volume without requiring a full platform rewrite

## Practical Interpretation Of “Scale”

Scaling to ~15,000 PDFs does **not** mean:

- instant full-corpus reindexing
- always-on GPU inference
- production-grade cloud infrastructure

Scaling to ~15,000 PDFs **does** mean:

- incremental updates
- careful metadata handling
- compact operational workflows
- CPU-tolerant retrieval and indexing decisions
- avoiding expensive architectural assumptions early

## Monthly Cost Analysis

### Cost target

The system must remain operable under a hard budget cap of **$30/month**.

This means the practical hosting target is:

- local workstation or lab desktop when possible, or
- one cheap CPU VPS running both the API and Qdrant server

Managed vector infrastructure should be treated as optional, not the default.

### Current observed footprint

Based on the production deployment as of 2026-03-19:

- 182 PDFs in the corpus (4 duplicate local, 178 unique indexed)
- 12,146 chunks in Qdrant
- about 37 MB of Qdrant storage (estimated; scales linearly)
- about 470 MB of source PDFs

### Straight-line scale estimate for ~15,000 PDFs

If chunk density stays similar, a rough projection is:

- about 415,000 chunks
- about 3.1 GB of Qdrant storage
- about 38 GB of source PDFs

This is only a planning estimate. Real size will depend on:

- average paper length
- chunking strategy
- duplicate rate
- metadata payload size

In practice, disk planning should assume at least **60 to 80 GB** to leave room for:

- source PDFs
- Qdrant storage growth
- manifests
- logs
- rebuild headroom

### Current deployment

The system is running on an **IONOS VPS** (CPU-only, Ubuntu 24.04). Stack:
- Docker: docker.io + docker-compose v1.29.2
- Python: 3.12, venv at `.venv/`
- Qdrant: server mode via docker-compose, data in `labvdb_qdrant_data` Docker volume
- API: uvicorn on port 8000, public-facing, no auth (single user)
- Initial corpus index complete 2026-03-19: 178 docs, 12,146 chunks
- Measured latency: p50=30.2ms, p95=38.2ms

### Deployment cost options

#### Option 1: local workstation

Recurring infrastructure cost:

- effectively $0/month if the lab already owns the machine

Estimated electricity cost:

- roughly $2 to $8/month for a machine averaging about 15 W to 60 W continuously
- this estimate uses the U.S. average residential electricity price of **17.24 cents/kWh** from EIA data for December 2025

#### Option 2: cheap self-hosted CPU VPS (current approach)

The system currently runs on an IONOS VPS. Other comparable providers include:

Hetzner Cloud:

- 4 GB RAM / 40 GB SSD: about **$4.09/month**
- 8 GB RAM / 80 GB SSD: about **$6.59/month**
- 16 GB RAM / 160 GB SSD: about **$10.59/month**

DigitalOcean basic droplets:

- 1 GB RAM / 25 GB SSD: **$6/month**
- 2 GB RAM / 50 GB SSD: **$12/month**
- 4 GB RAM / 80 GB SSD: **$24/month**
- 8 GB RAM / 160 GB SSD: **$48/month**

Interpretation:

- current corpus size is easily viable on a low-end CPU VPS
- ~15,000 PDFs is still viable on a single modest CPU VPS if indexing remains batch-oriented

### Managed vector hosting

Qdrant Cloud currently offers:

- a 1 GB free forever cluster
- paid usage-based managed hosting above that level

Implication:

- current prototype size may fit within the free tier
- the projected ~15,000 PDF corpus will likely exceed 1 GB and should not assume free-tier viability
- managed Qdrant should not be treated as the budget-default option unless pricing remains comfortably below self-hosted alternatives

### Embedding cost implications

If embeddings stay local using `BAAI/bge-small-en-v1.5`:

- monthly API embedding cost is effectively **$0**
- the tradeoff is CPU time during indexing and search

If hosted embeddings are used later, for example OpenAI:

- `text-embedding-3-small` pricing is low enough that embedding cost would likely be a small one-time or occasional batch cost rather than a major monthly driver

Therefore:

- embedding API cost is not the primary budget risk
- always-on managed infrastructure is the real budget risk

### Budget conclusion

The project remains financially viable under the $30/month constraint if it follows this operating model:

1. prefer local or single-node CPU deployment
2. keep indexing incremental and batch-oriented
3. avoid always-on GPU dependencies
4. avoid managed infrastructure unless its pricing is clearly below the self-hosted CPU VPS path

### Current hosting

The system is deployed on an **IONOS VPS** (CPU-only). This is already within the $30/month budget ceiling and leaves room for incidental backup or storage overhead.
