from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
import uvicorn

from labvdb import (
    COLLECTION_NAME,
    MODEL_NAME,
    build_filter,
    count_indexed_docs,
    count_unique_doc_ids,  # retained for reconcile; not called during normal operation
    ensure_collection,
    fetch_chunk_text,
    get_client,
    load_embedding_model,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = get_client()
    ensure_collection(client)
    app.state.client = client
    yield
    # Nothing to explicitly close for either embedded or server client.


app = FastAPI(title="PDF Search", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    request.app.state.client.get_collections()
    return {"status": "ok"}


@app.get("/chunk/{chunk_id}")
def chunk(chunk_id: str):
    text = fetch_chunk_text(chunk_id)
    if text is None:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return {"chunk_id": chunk_id, "text": text}


@app.get("/search")
def search(
    request: Request,
    q: str = Query(..., description="Search query", min_length=1),
    limit: int = Query(10, ge=1, le=50),
    doc_id: str | None = Query(None, description="Restrict results to one document"),
    filename: str | None = Query(None, description="Restrict results to one filename"),
):
    client = request.app.state.client
    model = load_embedding_model(MODEL_NAME)
    query_vector = model.encode(q, normalize_embeddings=True).tolist()
    search_filter = build_filter(doc_id=doc_id, filename=filename)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        limit=limit,
    ).points

    formatted_results = []
    for hit in results:
        formatted_results.append(
            {
                "score": hit.score,
                "doc_id": hit.payload["doc_id"],
                "filename": hit.payload["filename"],
                "page": hit.payload["page"],
                "chunk_idx": hit.payload["chunk_idx"],
                "chunk_id": str(hit.id),
                "preview": hit.payload.get("preview", ""),
            }
        )

    return {
        "query": q,
        "filters": {"doc_id": doc_id, "filename": filename},
        "results": formatted_results,
    }


@app.get("/stats")
def stats(request: Request):
    client = request.app.state.client
    point_count = client.count(collection_name=COLLECTION_NAME, exact=False).count
    return {
        "collection": COLLECTION_NAME,
        "point_count": point_count,
        "document_count": count_indexed_docs(),
    }


@app.get("/", response_class=HTMLResponse)
def ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PDF Search</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 960px; margin: 50px auto; padding: 20px; }
            h1 { color: #222; }
            .controls { display: grid; gap: 12px; margin-bottom: 20px; }
            .row { display: flex; gap: 10px; }
            input[type="text"] { flex: 1; padding: 12px; font-size: 16px; border: 2px solid #ddd; border-radius: 4px; }
            button { padding: 12px 24px; font-size: 16px; background: #0a66c2; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #084f96; }
            .result { background: #f8f9fa; padding: 15px; margin: 15px 0; border-left: 4px solid #0a66c2; border-radius: 4px; }
            .filename { font-weight: bold; color: #0a66c2; }
            .meta { color: #666; font-size: 14px; margin-top: 4px; }
            .text { margin-top: 8px; color: #333; }
            .score { float: right; background: #1f8f4e; color: white; padding: 4px 8px; border-radius: 3px; font-size: 12px; }
            .no-results { text-align: center; color: #666; padding: 40px; }
            .stats { color: #555; font-size: 14px; margin-bottom: 16px; }
        </style>
    </head>
    <body>
        <h1>PDF Search</h1>
        <div id="stats" class="stats"></div>
        <div class="controls">
            <div class="row">
                <input type="text" id="query" placeholder="Search for anything..." onkeypress="if(event.key==='Enter') search()">
                <button onclick="search()">Search</button>
            </div>
            <div class="row">
                <input type="text" id="doc_id" placeholder="Optional doc_id filter">
                <input type="text" id="filename" placeholder="Optional filename filter">
            </div>
        </div>
        <div id="results"></div>

        <script>
            async function loadStats() {
                const response = await fetch('/stats');
                const data = await response.json();
                document.getElementById('stats').textContent = `Indexed documents: ${data.document_count} | Indexed chunks: ${data.point_count}`;
            }

            async function search() {
                const query = document.getElementById('query').value.trim();
                const docId = document.getElementById('doc_id').value.trim();
                const filename = document.getElementById('filename').value.trim();
                if (!query) return;

                const params = new URLSearchParams({ q: query });
                if (docId) params.set('doc_id', docId);
                if (filename) params.set('filename', filename);

                const response = await fetch(`/search?${params.toString()}`);
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
                        <div class="meta">doc_id ${r.doc_id} | page ${r.page} | chunk ${r.chunk_idx}</div>
                        <div class="text">${r.text}</div>
                    </div>
                `).join('');
            }

            loadStats();
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
