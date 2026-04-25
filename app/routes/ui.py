"""Simple web UI route for quick manual upload/query usage."""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["UI"])

_HOME_PAGE_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>RAGify</title>
    <style>
      *, *::before, *::after { box-sizing: border-box; }
      body { font-family: Arial, sans-serif; margin: 0; padding: 2rem; max-width: 780px; background: #f9f9fb; }
      h1 { margin-bottom: 0.25rem; color: #1a1a2e; }
      .subtitle { color: #555; font-size: 0.95rem; margin-bottom: 2rem; }
      .card { background: #fff; border: 1px solid #dde; border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
      .card h2 { margin-top: 0; font-size: 1.1rem; color: #222; }
      label { display: block; margin-bottom: 0.4rem; font-weight: 600; font-size: 0.95rem; }
      input[type=file], input[type=text] { font-size: 1rem; width: 100%; padding: 0.4rem 0.6rem; border: 1px solid #ccc; border-radius: 6px; }
      button { margin-top: 0.75rem; padding: 0.5rem 1.2rem; font-size: 1rem; background: #4361ee; color: #fff; border: none; border-radius: 6px; cursor: pointer; }
      button:hover { background: #3451d1; }
      button:disabled { background: #999; cursor: not-allowed; }
      .result-box { margin-top: 1rem; padding: 1rem; background: #f0f4ff; border: 1px solid #c8d4ff; border-radius: 6px; display: none; }
      .result-box.error { background: #fff0f0; border-color: #ffb3b3; }
      .result-box.success { background: #f0fff4; border-color: #b3ffcc; }
      .result-box pre { margin: 0; white-space: pre-wrap; word-break: break-word; font-size: 0.92rem; }
      .answer-text { font-size: 1rem; line-height: 1.55; }
      .sources { margin-top: 0.75rem; font-size: 0.88rem; color: #444; }
      .sources strong { display: block; margin-bottom: 0.2rem; }
      .meta { font-size: 0.82rem; color: #777; margin-top: 0.5rem; }
      a { color: #4361ee; }
    </style>
  </head>
  <body>
    <h1>RAGify</h1>
    <p class="subtitle">Upload documents and query them with an AI. &nbsp;|&nbsp; <a href="/docs">/docs</a> for API reference.</p>

    <!-- Upload card -->
    <div class="card">
      <h2>📄 Upload &amp; Ingest Documents</h2>
      <label for="files">Select files (PDF, TXT, DOCX)</label>
      <input id="files" name="files" type="file" accept=".pdf,.txt,.docx" multiple />
      <button id="upload-btn" onclick="uploadFiles()">Upload &amp; Ingest</button>
      <div id="upload-result" class="result-box"></div>
    </div>

    <!-- Query card -->
    <div class="card">
      <h2>💬 Ask a Question</h2>
      <label for="q">Your question</label>
      <input id="q" type="text" placeholder="e.g. What are the main findings?" />
      <button id="query-btn" onclick="queryDocuments()">Ask</button>
      <div id="query-result" class="result-box"></div>
    </div>

    <script>
      async function uploadFiles() {
        const input = document.getElementById('files');
        const resultBox = document.getElementById('upload-result');
        const btn = document.getElementById('upload-btn');
        if (!input.files.length) {
          showResult(resultBox, 'error', 'Please select at least one file.');
          return;
        }
        btn.disabled = true;
        btn.textContent = 'Uploading…';
        const form = new FormData();
        for (const f of input.files) form.append('files', f);
        try {
          const res = await fetch('/ingest/', { method: 'POST', body: form });
          const data = await res.json();
          if (res.ok) {
            showResult(resultBox, 'success',
              `✅ ${data.message}\\n` +
              `Files processed: ${data.files_processed}  |  Chunks indexed: ${data.chunks_indexed}`
            );
          } else {
            const detail = data.detail?.message || JSON.stringify(data.detail || data);
            showResult(resultBox, 'error', '❌ ' + detail);
          }
        } catch (e) {
          showResult(resultBox, 'error', '❌ Network error: ' + e.message);
        } finally {
          btn.disabled = false;
          btn.textContent = 'Upload & Ingest';
        }
      }

      async function queryDocuments() {
        const q = document.getElementById('q').value.trim();
        const resultBox = document.getElementById('query-result');
        const btn = document.getElementById('query-btn');
        if (!q) { showResult(resultBox, 'error', 'Please enter a question.'); return; }
        btn.disabled = true;
        btn.textContent = 'Thinking…';
        try {
          const res = await fetch('/query/?q=' + encodeURIComponent(q));
          const data = await res.json();
          if (res.ok) {
            let html = '<div class="answer-text">' + escHtml(data.answer) + '</div>';
            if (data.sources && data.sources.length) {
              html += '<div class="sources"><strong>Sources:</strong>';
              data.sources.forEach(s => {
                html += escHtml(s.source);
                if (s.page != null) html += ' (page ' + s.page + ')';
                html += '<br/>';
              });
              html += '</div>';
            }
            html += '<div class="meta">Latency: ' + data.latency_ms + ' ms &nbsp;|&nbsp; Chunks retrieved: ' + data.retrieved_docs + '</div>';
            resultBox.innerHTML = html;
            resultBox.className = 'result-box success';
            resultBox.style.display = 'block';
          } else {
            const detail = data.detail || JSON.stringify(data);
            showResult(resultBox, 'error', '❌ ' + (typeof detail === 'string' ? detail : JSON.stringify(detail)));
          }
        } catch (e) {
          showResult(resultBox, 'error', '❌ Network error: ' + e.message);
        } finally {
          btn.disabled = false;
          btn.textContent = 'Ask';
        }
      }

      document.getElementById('q').addEventListener('keydown', e => {
        if (e.key === 'Enter') queryDocuments();
      });

      function showResult(box, type, text) {
        box.innerHTML = '<pre>' + escHtml(text) + '</pre>';
        box.className = 'result-box ' + type;
        box.style.display = 'block';
      }

      function escHtml(str) {
        return String(str)
          .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
          .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
      }
    </script>
  </body>
</html>
"""


@router.get("/", response_class=HTMLResponse)
async def home() -> str:
    """Return a minimal HTML page with upload and query forms."""
    return _HOME_PAGE_HTML
