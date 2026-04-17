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
      body { font-family: Arial, sans-serif; margin: 2rem; max-width: 720px; }
      h1 { margin-bottom: 0.5rem; }
      form { margin: 1rem 0 2rem; padding: 1rem; border: 1px solid #ddd; border-radius: 8px; }
      label { display: block; margin-bottom: 0.5rem; font-weight: 600; }
      input, button { font-size: 1rem; }
      .hint { color: #555; font-size: 0.95rem; }
    </style>
  </head>
  <body>
    <h1>RAGify</h1>
    <p class="hint">Use this page for quick testing, or open <a href="/docs">/docs</a> for interactive API docs.</p>

    <form action="/ingest/" method="post" enctype="multipart/form-data">
      <label for="files">Upload documents (PDF, TXT, DOCX)</label>
      <input id="files" name="files" type="file" aria-label="Upload documents" multiple required />
      <button type="submit">Upload & Ingest</button>
    </form>

    <form action="/query/" method="get">
      <label for="q">Ask a question</label>
      <input id="q" name="q" type="text" placeholder="Type your question" required style="width: 100%; max-width: 520px;" />
      <div style="margin-top: 0.75rem;">
        <button type="submit">Query</button>
      </div>
    </form>
  </body>
</html>
"""


@router.get("/", response_class=HTMLResponse)
async def home() -> str:
    """Return a minimal HTML page with upload and query forms."""
    return _HOME_PAGE_HTML
