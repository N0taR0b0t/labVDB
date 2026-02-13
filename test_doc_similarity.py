import pymupdf as fitz
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np

PDF_DIR = Path("pdfs")
SIM_THRESHOLD = 0.995  # strict duplicate threshold

print("Loading embedding model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
print("Model loaded!\n")

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
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Load PDFs
pdf_files = sorted(PDF_DIR.glob("*.pdf"), key=lambda p: p.name.lower())

if not pdf_files:
    print("No PDFs found.")
    exit(1)

print(f"Found {len(pdf_files)} PDFs\n")

# Extract text + embed
doc_embeddings = {}
doc_lengths = {}

for pdf in pdf_files:
    print(f"Processing: {pdf.name}")
    text = extract_full_text(pdf)
    doc_lengths[pdf.name] = len(text)
    embedding = model.encode(text)
    doc_embeddings[pdf.name] = embedding

print("\n--- Similarity Results ---\n")

checked_pairs = set()
duplicates_found = False

for i, name1 in enumerate(doc_embeddings):
    for j, name2 in enumerate(doc_embeddings):
        if j <= i:
            continue

        pair = tuple(sorted((name1, name2)))
        if pair in checked_pairs:
            continue

        sim = cosine_similarity(
            doc_embeddings[name1],
            doc_embeddings[name2]
        )

        if sim >= SIM_THRESHOLD:
            duplicates_found = True
            print(f"⚠️  POSSIBLE DUPLICATE:")
            print(f"    {name1}")
            print(f"    {name2}")
            print(f"    Similarity: {sim:.6f}")
            print()

        checked_pairs.add(pair)

if not duplicates_found:
    print("No high-similarity duplicates detected.")

print("\nDone.")
