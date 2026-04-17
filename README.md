# RAGify

A **production-grade Retrieval-Augmented Generation (RAG)** system built with Python, FastAPI, LangChain, SentenceTransformers, and FAISS. Upload your documents (PDF, TXT, DOCX) and query them with an LLM to get accurate, context-aware answers with source citations.

---

## 📁 Project Structure

```
RAGify/
├── app/
│   ├── main.py               # FastAPI entrypoint & lifespan
│   ├── routes/
│   │   ├── health.py         # GET  /health/
│   │   ├── ingest.py         # POST /ingest/
│   │   └── query.py          # GET  /query/
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
│   ├── generator.py          # OpenAI-backed answer generator
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

## ⚡ Quick Start

### Prerequisites

* Python 3.10+
* An OpenAI API key (optional – works without one, but answers will be stubs)

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
# Edit .env and set OPENAI_API_KEY (and other values as needed)
```

### 3 · Run the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

After startup:

* Home page (with upload form): <http://localhost:8000/>
* Interactive API docs (Swagger): <http://localhost:8000/docs>

### 4 · Upload files and ask questions

You can use either the **home page** or **Swagger UI**.

#### Option A: Home page (easy)

1. Open <http://localhost:8000/>
2. Use **Upload documents** and select one or more files (`.pdf`, `.txt`, `.docx`)
3. Click **Upload & Ingest**
4. Use the **Ask a question** box and submit your query

#### Option B: Swagger UI

1. Open <http://localhost:8000/docs>
2. Expand `POST /ingest/` and click **Try it out**
3. Click **Choose Files**, select files, then **Execute**
4. Expand `GET /query/`, provide `q`, then **Execute**

---

## 🐳 Docker

```bash
# Build
docker build -t ragify .

# Run
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

| Variable               | Default               | Description                         |
|------------------------|-----------------------|-------------------------------------|
| `OPENAI_API_KEY`       | *(empty)*             | OpenAI API key                      |
| `OPENAI_MODEL`         | `gpt-3.5-turbo`       | Chat model to use                   |
| `OPENAI_MAX_TOKENS`    | `512`                 | Max tokens in generated answer      |
| `OPENAI_TEMPERATURE`   | `0.2`                 | Sampling temperature                |
| `EMBEDDING_MODEL`      | `all-MiniLM-L6-v2`   | SentenceTransformer model           |
| `EMBEDDING_CACHE_DIR`  | `./data/embedding_cache` | Model weight cache directory    |
| `FAISS_INDEX_PATH`     | `./data/faiss_index`  | FAISS index persistence path        |
| `TOP_K`                | `5`                   | Default retrieval top-k             |
| `CHUNK_SIZE`           | `500`                 | Characters per text chunk           |
| `CHUNK_OVERLAP`        | `100`                 | Overlap characters between chunks   |
| `LOG_LEVEL`            | `INFO`                | Logging verbosity                   |
| `QUERY_CACHE_SIZE`     | `128`                 | In-memory LRU query cache size      |

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
                   OpenAI ChatCompletion → answer + citations
```

---

## 🔥 Advanced Features

| Feature | Status | Details |
|---------|--------|---------|
| LRU query caching | ✅ | Bounded in-memory cache in `RAGPipeline` |
| Query rewriting | ✅ | `?rewrite=true` — LLM rewrites query before retrieval |
| Hybrid retrieval | ✅ | `?hybrid=true` — BM25 + dense vector weighted combination |
| Cross-encoder reranking | ✅ | Opt-in via `RAGPipeline(use_reranker=True)` |
| Response citations | ✅ | `sources` field in every `/query/` response |
| Structured prompts | ✅ | "Answer ONLY using the context below" system prompt |
| Persistent FAISS index | ✅ | Automatically saved/loaded between restarts |

---

## 🚀 Future Improvements

* **Redis caching** – replace in-memory LRU with a distributed Redis cache
* **Async embeddings** – run embedding in a background thread pool
* **Multi-tenant namespacing** – isolate documents by user/project
* **Streaming responses** – stream LLM tokens back to the client via SSE
* **Document metadata filters** – filter retrieval by document type or date
* **LLaMA / Mistral support** – swap OpenAI for a local model via `llama-cpp-python`
* **Streamlit UI** – simple front-end for non-technical users
* **Observability** – OpenTelemetry traces and Prometheus metrics

---

## 📄 License

MIT
