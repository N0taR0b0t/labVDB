from __future__ import annotations

import time
from contextlib import asynccontextmanager

import psutil
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
    manifest_stats,
    rerank_hybrid,
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


def _deduplicate(results: list[dict], max_per_doc: int) -> list[dict]:
    seen: dict[str, int] = {}
    out = []
    for item in results:
        doc_id = item["doc_id"]
        if seen.get(doc_id, 0) < max_per_doc:
            out.append(item)
            seen[doc_id] = seen.get(doc_id, 0) + 1
    return out


@app.get("/search")
def search(
    request: Request,
    q: str = Query(..., description="Search query", min_length=1),
    limit: int = Query(10, ge=1, le=50),
    doc_id: str | None = Query(None, description="Restrict results to one document"),
    filename: str | None = Query(None, description="Restrict results to one filename"),
    section: str | None = Query(None, description="Restrict results to one section"),
    hybrid: bool = Query(True, description="Use hybrid dense+lexical reranking"),
    max_per_doc: int = Query(2, ge=1, le=10, description="Max results returned per document"),
):
    client = request.app.state.client
    model = load_embedding_model(MODEL_NAME)
    query_vector = model.encode(q, normalize_embeddings=True).tolist()
    search_filter = build_filter(doc_id=doc_id, filename=filename, section=section)

    fetch_limit = limit * 5 if hybrid else limit * 2
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        limit=fetch_limit,
    ).points

    if hybrid:
        formatted_results = rerank_hybrid(q, results, fetch_limit)
    else:
        formatted_results = [
            {
                "score": hit.score,
                "doc_id": hit.payload["doc_id"],
                "filename": hit.payload["filename"],
                "page": hit.payload["page"],
                "chunk_idx": hit.payload["chunk_idx"],
                "section": hit.payload.get("section", "Unknown"),
                "chunk_id": str(hit.id),
                "preview": hit.payload.get("preview", ""),
            }
            for hit in results
        ]

    # Skip per-doc cap when the user has already filtered to a single document.
    if doc_id is None and filename is None:
        formatted_results = _deduplicate(formatted_results, max_per_doc)

    return {
        "query": q,
        "hybrid": hybrid,
        "filters": {"doc_id": doc_id, "filename": filename, "section": section},
        "results": formatted_results[:limit],
    }


@app.get("/stats")
def stats(request: Request):
    client = request.app.state.client
    point_count = client.count(collection_name=COLLECTION_NAME, exact=False).count
    return {
        "collection": COLLECTION_NAME,
        "point_count": point_count,
        "document_count": count_indexed_docs(),
        "manifest": manifest_stats(),
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
            .text { margin-top: 8px; color: #333; cursor: pointer; }
            .text.expanded { cursor: default; white-space: pre-wrap; }
            .expand-hint { font-size: 12px; color: #0a66c2; margin-top: 4px; }
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
                <input type="text" id="section" placeholder="Optional section filter (e.g. Methods)">
            </div>
        </div>
        <div id="results"></div>

        <script>
            async function loadStats() {
                const response = await fetch('/stats');
                const data = await response.json();
                document.getElementById('stats').textContent = `Indexed documents: ${data.document_count} | Indexed chunks: ${data.point_count}`;
            }

            async function expandChunk(el, chunkId) {
                if (el.classList.contains('expanded')) return;
                el.classList.add('expanded');
                el.onclick = null;
                const hint = el.querySelector('.expand-hint');
                if (hint) hint.textContent = 'Loading...';
                try {
                    const resp = await fetch('/chunk/' + encodeURIComponent(chunkId));
                    if (!resp.ok) throw new Error('HTTP ' + resp.status);
                    const data = await resp.json();
                    el.querySelector('.preview-text').textContent = data.text;
                    if (hint) hint.remove();
                } catch (err) {
                    if (hint) hint.textContent = 'Failed to load (' + err.message + ')';
                    el.classList.remove('expanded');
                    el.onclick = () => expandChunk(el, chunkId);
                }
            }

            async function search() {
                const query = document.getElementById('query').value.trim();
                const docId = document.getElementById('doc_id').value.trim();
                const filename = document.getElementById('filename').value.trim();
                const section = document.getElementById('section').value.trim();
                if (!query) return;

                const params = new URLSearchParams({ q: query });
                if (docId) params.set('doc_id', docId);
                if (filename) params.set('filename', filename);
                if (section) params.set('section', section);

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
                        <div class="meta">page ${r.page} | section ${r.section} | chunk ${r.chunk_idx}</div>
                        <div class="text" onclick="expandChunk(this, '${r.chunk_id}')">
                            <span class="preview-text">${r.preview}</span>
                            <div class="expand-hint">Click to expand full text</div>
                        </div>
                    </div>
                `).join('');
            }

            loadStats();
        </script>
    </body>
    </html>
    """


@app.get("/metrics")
def metrics():
    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    return {
        "timestamp": time.time(),
        "cpu_percent": round(cpu, 1),
        "memory_percent": round(mem.percent, 1),
        "memory_used_mb": mem.used // (1024 * 1024),
        "memory_total_mb": mem.total // (1024 * 1024),
    }


@app.get("/monitor", response_class=HTMLResponse)
def monitor():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>System Monitor</title>
        <style>
            html, body { height: 100%; margin: 0; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 960px; margin: 0 auto; padding: 20px 20px 16px; box-sizing: border-box; display: flex; flex-direction: column; background: #111827; color: #e5e7eb; }
            h1 { color: #f9fafb; margin: 0 0 12px; flex-shrink: 0; }
            a { color: #60a5fa; text-decoration: none; font-size: 14px; }
            a:hover { text-decoration: underline; }
            .toolbar { display: flex; align-items: center; gap: 20px; margin-bottom: 16px; flex-shrink: 0; }
            select { padding: 6px 10px; border: 1px solid #374151; border-radius: 4px; font-size: 14px; cursor: pointer; background: #1f2937; color: #e5e7eb; }
            .toggle { font-size: 14px; color: #9ca3af; cursor: pointer; user-select: none; display: flex; align-items: center; gap: 6px; }
            .toggle input { cursor: pointer; accent-color: #60a5fa; width: 14px; height: 14px; }
            .metric { flex: 1; display: flex; flex-direction: column; min-height: 0; margin-bottom: 12px; }
            .metric:last-child { margin-bottom: 0; }
            .metric-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; flex-shrink: 0; }
            .metric-label { font-weight: 600; font-size: 15px; color: #9ca3af; }
            .metric-right { text-align: right; }
            .metric-val { font-size: 28px; font-weight: 700; font-variant-numeric: tabular-nums; }
            .metric-sub { font-size: 13px; color: #6b7280; margin-left: 6px; font-weight: 400; }
            #cpu-val { color: #60a5fa; }
            #mem-val { color: #34d399; }
            canvas { display: block; width: 100%; flex: 1; min-height: 0; border-radius: 4px; }
        </style>
    </head>
    <body>
        <h1>System Monitor</h1>
        <div class="toolbar">
            <label>Window:&nbsp;<select id="window-select">
                <option value="120000">2 min</option>
                <option value="300000">5 min</option>
                <option value="600000" selected>10 min</option>
                <option value="1200000">20 min</option>
            </select></label>
            <label class="toggle"><input type="checkbox" id="cpu-smooth" checked> Avg CPU</label>
            <a href="/">\u2190 Search</a>
        </div>

        <div class="metric">
            <div class="metric-header">
                <span class="metric-label">CPU Usage</span>
                <div class="metric-right">
                    <span id="cpu-val" class="metric-val">\u2014</span>
                </div>
            </div>
            <canvas id="cpu-canvas"></canvas>
        </div>

        <div class="metric">
            <div class="metric-header">
                <span class="metric-label">Memory</span>
                <div class="metric-right">
                    <span id="mem-val" class="metric-val">\u2014</span>
                    <span id="mem-sub" class="metric-sub"></span>
                </div>
            </div>
            <canvas id="mem-canvas"></canvas>
        </div>

        <script>
            const STORAGE_KEY = 'labvdb_monitor';
            const MAX_AGE_MS  = 20 * 60 * 1000;

            function loadData() {
                try {
                    const raw = localStorage.getItem(STORAGE_KEY);
                    if (!raw) return [];
                    const cutoff = Date.now() - MAX_AGE_MS;
                    return JSON.parse(raw).filter(p => p.ts >= cutoff);
                } catch (e) { return []; }
            }

            function saveData() {
                try { localStorage.setItem(STORAGE_KEY, JSON.stringify(data)); }
                catch (e) {}
            }

            const data = loadData();
            let windowMs = 600000;
            let cpuSmooth = true;

            function nextIntervalMs(cpuPct) {
                return Math.round(2000 + (cpuPct / 100) * 4000);
            }

            async function poll() {
                try {
                    const resp = await fetch('/metrics');
                    if (!resp.ok) throw new Error('HTTP ' + resp.status);
                    const d = await resp.json();
                    const pt = {
                        ts: d.timestamp * 1000,
                        cpu: d.cpu_percent,
                        mem: d.memory_percent,
                        memUsed: d.memory_used_mb,
                        memTotal: d.memory_total_mb,
                    };
                    data.push(pt);
                    const cutoff = Date.now() - MAX_AGE_MS;
                    while (data.length && data[0].ts < cutoff) data.shift();
                    saveData();

                    document.getElementById('cpu-val').textContent = pt.cpu.toFixed(1) + '%';
                    document.getElementById('mem-val').textContent = pt.mem.toFixed(1) + '%';
                    document.getElementById('mem-sub').textContent =
                        '(' + pt.memUsed.toLocaleString() + '\u202f/\u202f' + pt.memTotal.toLocaleString() + '\u202fMB)';

                    redraw();
                    setTimeout(poll, nextIntervalMs(pt.cpu));
                } catch (e) {
                    setTimeout(poll, 5000);
                }
            }

            function niceScale(maxVal) {
                const steps = [1, 2, 5, 10, 20, 25, 50, 100];
                for (const step of steps) {
                    const yMax = Math.max(20, Math.ceil((maxVal * 1.15) / step) * step);
                    if (yMax / step <= 6) return { yMax, step };
                }
                return { yMax: 100, step: 20 };
            }

            function redraw() {
                drawChart('cpu-canvas', pt => pt.cpu, '#60a5fa', 'rgba(96,165,250,0.15)', cpuSmooth ? 5 : 1);
                drawChart('mem-canvas', pt => pt.mem, '#34d399', 'rgba(52,211,153,0.15)', 1);
            }

            function drawChart(canvasId, getValue, lineColor, fillColor, smoothN) {
                const canvas = document.getElementById(canvasId);
                if (!canvas.offsetWidth) return;
                canvas.width  = canvas.offsetWidth;
                canvas.height = canvas.offsetHeight;
                const W = canvas.width, H = canvas.height;
                const PL = 44, PR = 12, PT = 10, PB = 28;
                const cW = W - PL - PR, cH = H - PT - PB;

                const now = Date.now();
                const cutoff = now - windowMs;
                const pts = data.filter(p => p.ts >= cutoff);

                // Compute display values, optionally smoothed
                const raw = pts.map(getValue);
                const vals = smoothN > 1
                    ? raw.map((_, i) => {
                        const w = raw.slice(Math.max(0, i - smoothN + 1), i + 1);
                        return w.reduce((a, b) => a + b, 0) / w.length;
                      })
                    : raw;

                const { yMax, step } = niceScale(vals.length ? Math.max(...vals) : 0);

                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, W, H);

                ctx.fillStyle = '#1f2937';
                ctx.fillRect(PL, PT, cW, cH);

                const xOf = ts => PL + Math.max(0, Math.min(1, (ts - cutoff) / windowMs)) * cW;
                const yOf = v  => PT + cH - Math.max(0, Math.min(1, v / yMax)) * cH;

                ctx.font = '11px system-ui, sans-serif';

                // Horizontal grid lines + Y labels
                ctx.strokeStyle = '#374151';
                ctx.lineWidth = 1;
                ctx.textAlign = 'right';
                ctx.fillStyle = '#6b7280';
                for (let v = 0; v <= yMax; v += step) {
                    const y = Math.round(yOf(v)) + 0.5;
                    ctx.beginPath(); ctx.moveTo(PL, y); ctx.lineTo(PL + cW, y); ctx.stroke();
                    ctx.fillText(v + '%', PL - 5, y + 4);
                }

                // X axis time labels
                ctx.textAlign = 'center';
                ctx.fillStyle = '#6b7280';
                for (let i = 0; i <= 4; i++) {
                    const ts = cutoff + (i / 4) * windowMs;
                    const x  = Math.round(xOf(ts));
                    const dt = new Date(ts);
                    const lbl = dt.getHours().toString().padStart(2,'0') + ':' +
                                dt.getMinutes().toString().padStart(2,'0') + ':' +
                                dt.getSeconds().toString().padStart(2,'0');
                    ctx.fillText(lbl, x, PT + cH + 18);
                }

                // Border
                ctx.strokeStyle = '#374151';
                ctx.lineWidth = 1;
                ctx.strokeRect(PL + 0.5, PT + 0.5, cW, cH);

                if (!pts.length) return;

                ctx.save();
                ctx.beginPath(); ctx.rect(PL, PT, cW, cH); ctx.clip();

                if (pts.length === 1) {
                    ctx.fillStyle = lineColor;
                    ctx.beginPath();
                    ctx.arc(xOf(pts[0].ts), yOf(vals[0]), 3, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.restore();
                    return;
                }

                // Area fill
                ctx.beginPath();
                ctx.moveTo(xOf(pts[0].ts), yOf(vals[0]));
                for (let i = 1; i < pts.length; i++)
                    ctx.lineTo(xOf(pts[i].ts), yOf(vals[i]));
                ctx.lineTo(xOf(pts[pts.length-1].ts), PT + cH);
                ctx.lineTo(xOf(pts[0].ts), PT + cH);
                ctx.closePath();
                ctx.fillStyle = fillColor;
                ctx.fill();

                // Line
                ctx.beginPath();
                ctx.moveTo(xOf(pts[0].ts), yOf(vals[0]));
                for (let i = 1; i < pts.length; i++)
                    ctx.lineTo(xOf(pts[i].ts), yOf(vals[i]));
                ctx.strokeStyle = lineColor;
                ctx.lineWidth = 2;
                ctx.lineJoin = 'round';
                ctx.lineCap = 'round';
                ctx.stroke();

                // Latest value dot (ring)
                const lx = xOf(pts[pts.length-1].ts), ly = yOf(vals[vals.length-1]);
                ctx.fillStyle = lineColor;
                ctx.beginPath(); ctx.arc(lx, ly, 4, 0, Math.PI * 2); ctx.fill();
                ctx.fillStyle = '#1f2937';
                ctx.beginPath(); ctx.arc(lx, ly, 2, 0, Math.PI * 2); ctx.fill();

                ctx.restore();
            }

            document.getElementById('window-select').addEventListener('change', e => {
                windowMs = parseInt(e.target.value);
                redraw();
            });

            document.getElementById('cpu-smooth').addEventListener('change', e => {
                cpuSmooth = e.target.checked;
                redraw();
            });

            window.addEventListener('resize', redraw);

            if (data.length) {
                const last = data[data.length - 1];
                document.getElementById('cpu-val').textContent = last.cpu.toFixed(1) + '%';
                document.getElementById('mem-val').textContent = last.mem.toFixed(1) + '%';
                document.getElementById('mem-sub').textContent =
                    '(' + last.memUsed.toLocaleString() + '\u202f/\u202f' + last.memTotal.toLocaleString() + '\u202fMB)';
                requestAnimationFrame(redraw);
            }

            poll();
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
