# Persian Enterprise RAG

A local-first Persian RAG and administrative assistant built with FastAPI,
Qdrant, Ollama, hybrid retrieval, reranking, temporary document sessions,
SearXNG web search, and a dedicated HR analytics mode.

## Project layout

```text
.
├── src/persian_rag/       Application package and web templates
├── scripts/               Operational scripts such as document ingestion
├── config/                Versioned, non-secret application configuration
├── data/documents/        Local source documents (not committed)
├── data/hr/               Private HR datasets (not committed)
├── data/indexes/          Generated local indexes (not committed)
├── deploy/                Deployment assets and SearXNG configuration
├── tests/                 Automated tests
├── .env.example           Environment variable reference
├── pyproject.toml         Package, dependency, test, and lint configuration
└── requirements.txt       Simple pip-compatible dependency list
```

## Requirements

- Python 3.11+
- NVIDIA GPU with a compatible PyTorch installation (recommended)
- Ollama with `qwen3:8b`, `qwen3:14b`, and `qwen3-embedding:0.6b`
- Qdrant
- Docker Compose for the optional local SearXNG service

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Environment files are intentionally ignored by Git. Export the values from
`.env` through your process manager or shell before starting the application.

## Ingest documents

Place PDF files in `data/documents/`, then run:

```bash
python scripts/ingest.py
```

## Run

```bash
uvicorn persian_rag.main:app --app-dir src --host 127.0.0.1 --port 8000
```

The legacy entry point remains available after editable installation:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

## SearXNG

```bash
docker compose -f deploy/docker-compose.searxng.yml up -d
```

## Data privacy

HR spreadsheets, uploaded documents, generated indexes, secrets, and runtime
files are excluded from Git. The HR pipeline calculates statistics locally and
does not expose national IDs, insurance IDs, certificate IDs, or father names
to the language model.
