# RAGify

A **production-grade Retrieval-Augmented Generation (RAG)** system built with Python, FastAPI, LangChain, SentenceTransformers, and FAISS. Upload your documents (PDF, TXT, DOCX) and query them with an LLM to get accurate, context-aware answers with source citations.

> **No paid API key required.** The default configuration uses [Ollama](https://ollama.com) for completely free, local LLM inference. Groq and NVIDIA (free cloud APIs) and OpenAI are also supported.

---

## 📁 Project Structure

```
RAGify/
├── app/
│   ├── main.py               # FastAPI entrypoint & lifespan
│   ├── routes/
│   │   ├── health.py         # GET  /health/
│   │   ├── ingest.py         # POST /ingest/
│   │   ├── query.py          # GET  /query/
│   │   └── ui.py             # GET  / (web UI with file upload)
│   ├── services/
│   │   ├── ingestion_service.py
│   │   └── query_service.py
│   ├── core/
│   │   ├── logging.py
│   │   └── exceptions.py
│   └── utils/
│       └── schemas.py        # Pydantic request/response models
│
├── ingestion/
│   ├── loader.py             # PDF / TXT / DOCX loaders
│   ├── chunking.py           # Recursive text splitter
│   └── embedder.py           # SentenceTransformer wrapper
│
├── vectorstore/
│   └── faiss_store.py        # FAISS index with save/load/search
│
├── rag/
│   ├── retriever.py          # Dense + hybrid BM25 retrieval
│   ├── generator.py          # Multi-provider LLM answer generator
│   └── pipeline.py           # End-to-end RAG pipeline w/ caching
│
├── config/
│   └── settings.py           # Pydantic-Settings configuration
│
├── data/                     # Runtime data (git-ignored)
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_api.py
│
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── .env.example
└── README.md
```

---

## 🆓 Free LLM Options

RAGify supports four LLM providers. Select one with the `LLM_PROVIDER` environment variable.

### Option 1 – Ollama (completely free, local) ✅ *default*

Runs LLMs on your own machine — no API key, no cost, no data leaves your computer.

```bash
# 1. Install Ollama
#    macOS / Linux: https://ollama.com/download
#    Windows:       https://ollama.com/download/windows

# 2. Pull a model (choose one)
ollama pull llama2        # 3.8 GB – general purpose (default)
ollama pull mistral       # 4.1 GB – great quality
ollama pull phi3          # 2.3 GB – fast & lightweight
ollama pull gemma2        # 5.4 GB – Google Gemma 2

# 3. Start the server (usually auto-starts; run manually if needed)
ollama serve

# 4. Set in .env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama2       # or mistral, phi3, gemma2, etc.
```

### Option 2 – Groq (free cloud API)

Fast cloud inference with a **free tier** — no credit card needed.

1. Create a free account at <https://console.groq.com>
2. Generate an API key
3. Set in `.env`:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-8b-8192   # or mixtral-8x7b-32768, gemma-7b-it
```

### Option 3 – NVIDIA (free cloud API)

OpenAI-compatible cloud inference powered by NVIDIA NIM — free API key, no credit card needed.

1. Create a free account at <https://build.nvidia.com>
2. Generate an API key
3. Set in `.env`:

```env
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_MODEL=meta/llama3-70b-instruct   # or mistralai/mistral-7b-instruct-v0.3, etc.
```

### Option 4 – OpenAI (paid)

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
```

> **Note:** Embeddings always run locally via SentenceTransformers (`all-MiniLM-L6-v2`). No embedding API cost regardless of the LLM provider chosen.

---

## ⚡ Quick Start

### Prerequisites

* Python 3.10+
* One of the free LLM options above (Ollama recommended)

### 1 · Clone & install

```bash
git clone https://github.com/pankaj1281/RAGify.git
cd RAGify
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 · Configure environment

```bash
cp .env.example .env
# Edit .env – set LLM_PROVIDER and the matching credentials
# Default (ollama) works out of the box once Ollama is running
```

### 3 · Run the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

After startup:

* **Home page** (upload + query UI): <http://localhost:8000/>
* **Interactive API docs** (Swagger): <http://localhost:8000/docs>

### 4 · Upload files and ask questions

#### Option A: Home page (easy)

1. Open <http://localhost:8000/>
2. Click **"Select files"** and choose one or more files (`.pdf`, `.txt`, `.docx`)
3. Click **"Upload & Ingest"** – the result is shown inline on the page
4. Type a question in the **"Ask a Question"** box and press **Ask** or hit Enter
5. The answer and source citations appear instantly below

#### Option B: Swagger UI

1. Open <http://localhost:8000/docs>
2. Expand `POST /ingest/` and click **Try it out**
3. Click **Choose Files**, select files, then **Execute**
4. Expand `GET /query/`, provide `q`, then **Execute**

#### Option C: curl

```bash
# Upload a document
curl -X POST http://localhost:8000/ingest/ \
  -F "files=@my_document.pdf"

# Ask a question
curl "http://localhost:8000/query/?q=What+are+the+main+findings%3F"
```

---

## 🐳 Docker

```bash
# Build
docker build -t ragify .

# Run with Ollama (point to host Ollama server)
docker run -p 8000:8000 \
  -e LLM_PROVIDER=ollama \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434/v1 \
  ragify

# Run with Groq
docker run -p 8000:8000 \
  -e LLM_PROVIDER=groq \
  -e GROQ_API_KEY=your_key \
  ragify

# Run with .env file
docker run -p 8000:8000 --env-file .env ragify
```

---

## 🔌 API Reference

### `GET /health/`

Returns service health and the number of indexed document chunks.

```bash
curl http://localhost:8000/health/
```

```json
{
  "status": "ok",
  "version": "1.0.0",
  "indexed_documents": 42
}
```

---

### `POST /ingest/`

Upload one or more documents to index them in the vector store.

Supported formats: **PDF**, **TXT**, **DOCX**.

```bash
curl -X POST http://localhost:8000/ingest/ \
  -F "files=@my_document.pdf" \
  -F "files=@notes.txt"
```

```json
{
  "message": "Successfully ingested 2 file(s)",
  "files_processed": 2,
  "chunks_indexed": 84
}
```

---

### `GET /query/`

Ask a question against the indexed documents.

| Parameter | Type    | Default | Description                              |
|-----------|---------|---------|------------------------------------------|
| `q`       | string  | —       | **Required.** Natural-language question  |
| `k`       | integer | 5       | Number of context chunks to retrieve     |
| `rewrite` | boolean | false   | Rewrite query with LLM before retrieval  |
| `hybrid`  | boolean | false   | Use BM25 + vector hybrid retrieval       |

```bash
curl "http://localhost:8000/query/?q=What+are+the+main+findings%3F&k=5"
```

```json
{
  "question": "What are the main findings?",
  "answer": "The main findings are ...",
  "sources": [
    {"source": "report.pdf", "page": 3, "chunk_index": 7}
  ],
  "retrieved_docs": 5,
  "latency_ms": 430.5,
  "rewritten_query": "What are the main findings?"
}
```

---

## ⚙️ Configuration

All settings are controlled via environment variables (see `.env.example`):

| Variable               | Default                    | Description                                      |
|------------------------|----------------------------|--------------------------------------------------|
| `LLM_PROVIDER`         | `ollama`                   | LLM backend: `ollama`, `groq`, `nvidia`, or `openai` |
| `OLLAMA_BASE_URL`      | `http://localhost:11434/v1`| Ollama API base URL                              |
| `OLLAMA_MODEL`         | `llama2`                   | Ollama model name                                |
| `GROQ_API_KEY`         | *(empty)*                  | Groq API key (free at console.groq.com)          |
| `GROQ_MODEL`           | `llama3-8b-8192`           | Groq model name                                  |
| `NVIDIA_API_KEY`       | *(empty)*                  | NVIDIA API key (free at build.nvidia.com)        |
| `NVIDIA_BASE_URL`      | `https://integrate.api.nvidia.com/v1` | NVIDIA NIM API base URL             |
| `NVIDIA_MODEL`         | `meta/llama3-70b-instruct` | NVIDIA model name                                |
| `OPENAI_API_KEY`       | *(empty)*                  | OpenAI API key (paid)                            |
| `OPENAI_MODEL`         | `gpt-3.5-turbo`            | OpenAI model name                                |
| `OPENAI_MAX_TOKENS`    | `512`                      | Max tokens in generated answer                   |
| `OPENAI_TEMPERATURE`   | `0.2`                      | Sampling temperature                             |
| `EMBEDDING_MODEL`      | `all-MiniLM-L6-v2`        | SentenceTransformer model (local, free)          |
| `EMBEDDING_CACHE_DIR`  | `./data/embedding_cache`   | Model weight cache directory                     |
| `FAISS_INDEX_PATH`     | `./data/faiss_index`       | FAISS index persistence path                     |
| `TOP_K`                | `5`                        | Default retrieval top-k                          |
| `CHUNK_SIZE`           | `500`                      | Characters per text chunk                        |
| `CHUNK_OVERLAP`        | `100`                      | Overlap characters between chunks                |
| `LOG_LEVEL`            | `INFO`                     | Logging verbosity                                |
| `QUERY_CACHE_SIZE`     | `128`                      | In-memory LRU query cache size                   |

---

## 🧪 Testing

```bash
pip install -r requirements-dev.txt
pytest
```

Tests cover:

* **Ingestion** – loader, chunking, embedder
* **Retrieval** – FAISS vector store, similarity search, save/load
* **API** – health, ingest, query endpoints (mocked services)

---

## 🏗 Architecture

```
User Request
     │
     ▼
FastAPI (app/main.py)
     │
     ├──► POST /ingest/
     │        │
     │        ▼
     │    IngestionService
     │        │
     │        ▼
     │    loader.py ──► chunking.py ──► embedder.py
     │                                      │
     │                                      ▼
     │                               FAISSVectorStore.add_documents()
     │                               FAISSVectorStore.save()
     │
     └──► GET /query/
               │
               ▼
           QueryService
               │
               ▼
           RAGPipeline
               │
               ├──► (optional) query rewriting via LLM
               │
               ├──► Retriever.retrieve() / hybrid_retrieve()
               │        │
               │        ▼
               │    FAISSVectorStore.similarity_search()
               │
               ├──► (optional) CrossEncoder reranking
               │
               └──► Generator.generate()
                        │
                        ▼
                    LLM Provider (Ollama / Groq / NVIDIA / OpenAI)
                        │
                        ▼
                    answer + citations
```

---

## 🔥 Advanced Features

| Feature | Status | Details |
|---------|--------|---------|
| Free LLM (Ollama) | ✅ | Local inference, no API key, no cost |
| Free LLM (Groq) | ✅ | Fast cloud inference, free API key |
| Free LLM (NVIDIA) | ✅ | NVIDIA NIM cloud inference, free API key |
| LRU query caching | ✅ | Bounded in-memory cache in `RAGPipeline` |
| Query rewriting | ✅ | `?rewrite=true` — LLM rewrites query before retrieval |
| Hybrid retrieval | ✅ | `?hybrid=true` — BM25 + dense vector weighted combination |
| Cross-encoder reranking | ✅ | Opt-in via `RAGPipeline(use_reranker=True)` |
| Response citations | ✅ | `sources` field in every `/query/` response |
| Structured prompts | ✅ | "Answer ONLY using the context below" system prompt |
| Persistent FAISS index | ✅ | Automatically saved/loaded between restarts |
| File upload UI | ✅ | Home page with inline upload result & query answer |

---

## 🚀 Future Improvements

* **Redis caching** – replace in-memory LRU with a distributed Redis cache
* **Async embeddings** – run embedding in a background thread pool
* **Multi-tenant namespacing** – isolate documents by user/project
* **Streaming responses** – stream LLM tokens back to the client via SSE
* **Document metadata filters** – filter retrieval by document type or date
* **Streamlit UI** – simple front-end for non-technical users
* **Observability** – OpenTelemetry traces and Prometheus metrics

---

## 📄 License

MIT
