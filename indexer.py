import pymupdf as fitz
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path
import hashlib
import datetime
import numpy as np
import time
import json

# Initialize
print("Loading embedding model...")
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
print("Model loaded!")

# Qdrant in local persistent storage (FIXED)
client = QdrantClient(path="qdrant_storage")  # Remove the ./

# Create collection
collection_name = "pdfs"
RESET_COLLECTION = os.getenv("RESET_COLLECTION", "0") == "1"

BLACKLIST_PATH = Path("pdfs_blacklist.txt")
MANIFEST_PATH = Path("indexer_manifest.json")
SIM_THRESHOLD = 0.995  # strict duplicate threshold

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def sha1_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk_bytes), b""):
            h.update(block)
    return h.hexdigest()

def extract_full_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        txt = page.get_text()
        if txt:
            texts.append(txt)
    doc.close()
    return "\n".join(texts)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def choose_file_to_blacklist(file1: Path, file2: Path) -> Path:
    # Deterministic rule requested: blacklist alphabetically first, keep second.
    ordered = sorted([file1, file2], key=lambda p: p.name.lower())
    return ordered[0]

def load_blacklist(path: Path) -> set[str]:
    if not path.exists():
        return set()
    entries: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        entries.add(stripped)
    return entries

def save_blacklist(path: Path, entries: set[str]) -> None:
    header = [
        "# PDFs to ignore during indexing (one path per line).",
        "# Auto-updated by indexer.py when duplicates are detected.",
        f"# Updated: {datetime.datetime.now(datetime.UTC).isoformat()}",
        "",
    ]
    body = sorted(entries, key=lambda s: s.lower())
    path.write_text("\n".join(header + body) + "\n", encoding="utf-8")

def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "files": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "files": {}}
    if not isinstance(data, dict):
        return {"version": 1, "files": {}}
    if not isinstance(data.get("files"), dict):
        data["files"] = {}
    if "version" not in data:
        data["version"] = 1
    return data

def save_manifest(path: Path, state: dict) -> None:
    path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )

def doc_id_exists(doc_id: str) -> bool:
    flt = Filter(
        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
    )
    points, _next = client.scroll(
        collection_name=collection_name,
        scroll_filter=flt,
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return len(points) > 0

def make_point_id(doc_id: str, page_num: int, chunk_idx: int) -> int:
    # Deterministic per-chunk ID; keep within signed 64-bit range.
    digest = hashlib.md5(f"{doc_id}:{page_num}:{chunk_idx}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**63 - 1)

def index_pdf(pdf_path: str, doc_id: str):
    """Extract text from PDF, chunk it, and index"""
    print(f"\nIndexing: {pdf_path}")
    doc = fitz.open(pdf_path)
    filename = Path(pdf_path).name
    
    points = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        
        if not text.strip():
            continue
            
        chunks = chunk_text(text)
        
        for chunk_idx, chunk in enumerate(chunks):
            # Embed the chunk
            embedding = model.encode(chunk).tolist()
            
            # Create point
            points.append(PointStruct(
                id=make_point_id(doc_id=doc_id, page_num=page_num + 1, chunk_idx=chunk_idx),
                vector=embedding,
                payload={
                    "doc_id": doc_id,
                    "text": chunk,
                    "filename": filename,
                    "page": page_num + 1,
                    "chunk_idx": chunk_idx
                }
            ))
    
    # Upload to Qdrant
    if points:
        client.upsert(collection_name=collection_name, points=points)
        print(f"  ✓ Indexed {len(points)} chunks from {len(doc)} pages")
    
    doc.close()

# Index all PDFs
run_start = time.perf_counter()

manifest = load_manifest(MANIFEST_PATH)
manifest_files: dict[str, dict] = manifest["files"]

collection_fresh = False
if RESET_COLLECTION:
    try:
        client.delete_collection(collection_name=collection_name)
        print(f"Deleted existing collection '{collection_name}'")
    except Exception:
        pass
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print(f"Created fresh collection '{collection_name}'")
    collection_fresh = True
else:
    try:
        client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists")
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"Created collection '{collection_name}'")
        collection_fresh = True

if collection_fresh:
    manifest_files.clear()

pdf_dir = Path("pdfs")
pdf_files = sorted(pdf_dir.glob("*.pdf"), key=lambda p: p.name.lower())

if not pdf_files:
    print("No PDFs found in 'pdfs/' directory!")
    exit(1)

print(f"\nFound {len(pdf_files)} PDFs to index")

existing_blacklist = load_blacklist(BLACKLIST_PATH)
auto_blacklist: set[str] = set(existing_blacklist)

pdf_paths = {p.as_posix() for p in pdf_files}
stale_manifest_paths = [p for p in manifest_files.keys() if p not in pdf_paths]
for stale in stale_manifest_paths:
    del manifest_files[stale]

file_stats: dict[str, tuple[int, int]] = {}
candidate_files: list[Path] = []
candidate_paths: set[str] = set()
skipped_unchanged = 0
for pdf_file in pdf_files:
    pdf_path = pdf_file.as_posix()
    stat = pdf_file.stat()
    file_stats[pdf_path] = (int(stat.st_size), int(stat.st_mtime_ns))
    prev = manifest_files.get(pdf_path)
    unchanged = (
        isinstance(prev, dict)
        and prev.get("size") == file_stats[pdf_path][0]
        and prev.get("mtime_ns") == file_stats[pdf_path][1]
    )
    if unchanged and prev.get("status") in {"indexed", "skipped_doc_id_exists"}:
        skipped_unchanged += 1
        continue
    candidate_files.append(pdf_file)
    candidate_paths.add(pdf_path)

print(f"Unchanged files skipped from reprocessing: {skipped_unchanged}")
print(f"Files to process this run: {len(candidate_files)}")

# Semantic near-duplicate detection only for new/changed files.
doc_embeddings: dict[Path, np.ndarray] = {}
for pdf_file in candidate_files:
    if pdf_file.as_posix() in auto_blacklist:
        continue
    print(f"Computing doc embedding: {pdf_file.name}")
    full_text = extract_full_text(pdf_file)
    doc_embeddings[pdf_file] = model.encode(full_text)

for i, file1 in enumerate(candidate_files):
    if file1.as_posix() in auto_blacklist or file1 not in doc_embeddings:
        continue
    for j in range(i + 1, len(candidate_files)):
        file2 = candidate_files[j]
        if file2.as_posix() in auto_blacklist or file2 not in doc_embeddings:
            continue
        sim = cosine_similarity(doc_embeddings[file1], doc_embeddings[file2])
        if sim >= SIM_THRESHOLD:
            to_blacklist = choose_file_to_blacklist(file1, file2)
            kept = file2 if to_blacklist == file1 else file1
            print(
                f"Detected near-duplicate (sim={sim:.6f}): "
                f"keeping {kept.name}, blacklisting {to_blacklist.name}"
            )
            auto_blacklist.add(to_blacklist.as_posix())

if auto_blacklist != existing_blacklist:
    save_blacklist(BLACKLIST_PATH, auto_blacklist)
    newly_added = len(auto_blacklist) - len(existing_blacklist)
    if newly_added > 0:
        print(f"Added {newly_added} duplicate PDF(s) to {BLACKLIST_PATH}")

processed_files = 0
skipped_doc_id = 0

for pdf_file in pdf_files:
    try:
        pdf_path = pdf_file.as_posix()
        size, mtime_ns = file_stats[pdf_path]

        if pdf_file.as_posix() in auto_blacklist:
            print(f"\nSkipping blacklisted PDF: {pdf_file}")
            manifest_files[pdf_path] = {
                "status": "blacklisted",
                "size": size,
                "mtime_ns": mtime_ns,
                "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
            }
            continue

        if pdf_path not in candidate_paths:
            continue

        doc_id = sha1_file(pdf_file)
        if doc_id_exists(doc_id):
            print(f"\nSkipping already-indexed PDF (doc_id match): {pdf_file}")
            skipped_doc_id += 1
            manifest_files[pdf_path] = {
                "status": "skipped_doc_id_exists",
                "doc_id": doc_id,
                "size": size,
                "mtime_ns": mtime_ns,
                "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
            }
            continue
        index_pdf(str(pdf_file), doc_id=doc_id)
        processed_files += 1
        manifest_files[pdf_path] = {
            "status": "indexed",
            "doc_id": doc_id,
            "size": size,
            "mtime_ns": mtime_ns,
            "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
    except Exception as e:
        print(f"  ✗ Error indexing {pdf_file}: {e}")
        pdf_path = pdf_file.as_posix()
        size, mtime_ns = file_stats.get(pdf_path, (0, 0))
        manifest_files[pdf_path] = {
            "status": "error",
            "size": size,
            "mtime_ns": mtime_ns,
            "error": str(e),
            "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }

save_manifest(MANIFEST_PATH, manifest)
print("\n✅ Indexing complete!")
print(f"Total points in collection: {client.count(collection_name).count}")
elapsed = time.perf_counter() - run_start
time_per_file = elapsed / processed_files if processed_files > 0 else 0.0
print(f"Elapsed time (s): {elapsed:.2f}")
print(f"Files processed: {processed_files}")
print(f"Skipped (unchanged): {skipped_unchanged}")
print(f"Skipped (already indexed doc_id): {skipped_doc_id}")
print(f"Time per file (s): {time_per_file:.2f}")
