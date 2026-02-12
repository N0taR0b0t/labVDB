from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import uvicorn

# Initialize
print("Loading model...")
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
print("Connecting to Qdrant...")
client = QdrantClient(path="qdrant_storage")

app = FastAPI(title="PDF Search")

collection_name = "pdfs"

@app.get("/search")
def search(q: str = Query(..., description="Search query"), limit: int = 10):
    """Search PDFs by semantic similarity"""
    
    # Embed query
    query_vector = model.encode(q).tolist()
    
    # Search Qdrant - FIXED METHOD
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit
    ).points
    
    # Format results
    formatted_results = []
    for hit in results:
        formatted_results.append({
            "score": hit.score,
            "filename": hit.payload["filename"],
            "page": hit.payload["page"],
            "text": hit.payload["text"][:200] + "..." if len(hit.payload["text"]) > 200 else hit.payload["text"]
        })
    
    return {
        "query": q,
        "results": formatted_results
    }

@app.get("/", response_class=HTMLResponse)
def ui():
    """Simple search UI"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PDF Search</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 50px auto; padding: 20px; }
            h1 { color: #333; }
            input[type="text"] { width: 70%; padding: 12px; font-size: 16px; border: 2px solid #ddd; border-radius: 4px; }
            button { padding: 12px 24px; font-size: 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; margin-left: 10px; }
            button:hover { background: #0056b3; }
            .result { background: #f8f9fa; padding: 15px; margin: 15px 0; border-left: 4px solid #007bff; border-radius: 4px; }
            .filename { font-weight: bold; color: #007bff; }
            .page { color: #666; font-size: 14px; }
            .text { margin-top: 8px; color: #333; }
            .score { float: right; background: #28a745; color: white; padding: 4px 8px; border-radius: 3px; font-size: 12px; }
            .no-results { text-align: center; color: #666; padding: 40px; }
        </style>
    </head>
    <body>
        <h1>🔍 PDF Search</h1>
        <div>
            <input type="text" id="query" placeholder="Search for anything..." onkeypress="if(event.key==='Enter') search()">
            <button onclick="search()">Search</button>
        </div>
        <div id="results"></div>
        
        <script>
            async function search() {
                const query = document.getElementById('query').value;
                if (!query) return;
                
                const response = await fetch(`/search?q=${encodeURIComponent(query)}`);
                const data = await response.json();
                
                const resultsDiv = document.getElementById('results');
                
                if (data.results.length === 0) {
                    resultsDiv.innerHTML = '<div class="no-results">No results found</div>';
                    return;
                }
                
                resultsDiv.innerHTML = data.results.map(r => `
                    <div class="result">
                        <span class="score">Score: ${r.score.toFixed(3)}</span>
                        <div class="filename">${r.filename}</div>
                        <div class="page">Page ${r.page}</div>
                        <div class="text">${r.text}</div>
                    </div>
                `).join('');
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
