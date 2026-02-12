import pymupdf as fitz
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path
import hashlib

# Initialize
print("Loading embedding model...")
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
print("Model loaded!")

# Qdrant in local persistent storage (FIXED)
client = QdrantClient(path="qdrant_storage")  # Remove the ./

# Create collection
collection_name = "pdfs"

# Check if collection exists, if not create it
try:
    client.get_collection(collection_name)
    print(f"Collection '{collection_name}' already exists")
except:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print(f"Created collection '{collection_name}'")

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def index_pdf(pdf_path):
    """Extract text from PDF, chunk it, and index"""
    print(f"\nIndexing: {pdf_path}")
    doc = fitz.open(pdf_path)
    filename = Path(pdf_path).name
    
    points = []
    point_id = int(hashlib.md5(pdf_path.encode()).hexdigest()[:8], 16)
    
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
                id=point_id,
                vector=embedding,
                payload={
                    "text": chunk,
                    "filename": filename,
                    "page": page_num + 1,
                    "chunk_idx": chunk_idx
                }
            ))
            point_id += 1
    
    # Upload to Qdrant
    if points:
        client.upsert(collection_name=collection_name, points=points)
        print(f"  ✓ Indexed {len(points)} chunks from {len(doc)} pages")
    
    doc.close()

# Index all PDFs
pdf_dir = Path("pdfs")
pdf_files = list(pdf_dir.glob("*.pdf"))

if not pdf_files:
    print("No PDFs found in 'pdfs/' directory!")
    exit(1)

print(f"\nFound {len(pdf_files)} PDFs to index")

for pdf_file in pdf_files:
    try:
        index_pdf(str(pdf_file))
    except Exception as e:
        print(f"  ✗ Error indexing {pdf_file}: {e}")

print("\n✅ Indexing complete!")
print(f"Total points in collection: {client.count(collection_name).count}")
