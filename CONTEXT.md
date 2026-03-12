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

The chosen middle path is:

1. Keep one local Qdrant collection.
2. Use a lightweight local manifest for ingest state.
3. Use content-based `doc_id` values for deduplication and reindex safety.
4. Add payload indexes for fields used in filtering.
5. Improve chunking and cleanup before adding infrastructure complexity.
6. Move toward hybrid retrieval instead of relying on dense-only search.
7. Add basic observability so performance tradeoffs are measurable.

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
- Consider `bge-m3` only if CPU performance remains acceptable and the quality gain justifies the higher cost.
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

#### Critical (must complete before scaling)

**Phase 1 — Deploy server Qdrant with disk-backed operation**

Move from embedded local mode (`QdrantClient(path=...)`) to a standalone Qdrant server or Docker container (`QdrantClient(url=...)`). Configure on-disk vectors, on-disk HNSW, scalar quantization, and persistent SSD storage. Target: 8 GB RAM machine with SSD mandatory.

**Phase 2 — Make the manifest authoritative; eliminate whole-collection scans**

Treat `manifest.sqlite3` as the sole source of truth for which PDFs are known, whether a PDF changed, and whether it was indexed successfully. Remove Qdrant-wide scroll scans from the indexing planner and stats endpoint. Use `(path, size, mtime_ns, hash optional)` to detect file changes. A manual reconcile command should exist for repair, not as the normal code path.

**Phase 3 — Remove full chunk text from Qdrant payloads**

Store only lightweight search payload in Qdrant: `doc_id`, `filename`, `page`, `chunk_idx`, `section`, and a short preview. Store full chunk text in a separate SQLite chunk table keyed by `chunk_id`. The API fetches full text only for selected results when needed.

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

Based on the current repository state:

- 182 PDFs in the corpus
- 178 unique documents indexed
- 4,929 chunks in Qdrant
- about 37 MB of Qdrant storage
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

### Deployment cost options

#### Option 1: local workstation

Recurring infrastructure cost:

- effectively $0/month if the lab already owns the machine

Estimated electricity cost:

- roughly $2 to $8/month for a machine averaging about 15 W to 60 W continuously
- this estimate uses the U.S. average residential electricity price of **17.24 cents/kWh** from EIA data for December 2025

This is the cheapest path and remains the default recommendation for prototyping.

#### Option 2: cheap self-hosted CPU VPS

Representative March 11, 2026 pricing:

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
- Hetzner is substantially more compatible with the $30/month cap than DigitalOcean for this project

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

### Recommended hosting path

Near term:

- run locally on a lab machine when possible, or
- use one cheap Hetzner CPU VM

Best practical hosted target:

- **8 GB RAM / 80 GB SSD Hetzner Cloud** at about **$6.59/month**

Safer medium-term target for ~15,000 PDFs:

- **16 GB RAM / 160 GB SSD Hetzner Cloud** at about **$10.59/month**

Both remain well below the project’s $30/month ceiling and leave room for incidental backup or storage overhead.
