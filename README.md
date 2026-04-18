# RAG Research Assistant

Production-grade Retrieval-Augmented Generation system for research documents.

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://yourapp.streamlit.app)

## Architecture

```
PDF → PyMuPDF → Chunker → Sentence Transformers → ChromaDB
                                          ↓
Query → BM25 Search ─┬─→ RRF Fusion → Cross-Encoder Rerank → Groq LLM → Cited Answer
         Vector Search ─┘
```

## Key Features

1. **Hybrid Retrieval (BM25 + Vector)** - Catches exact keywords + semantic similarity
2. **Cross-Encoder Reranking** - ms-marco-MiniLM for precision scoring
3. **Citation Enforcement** - Every claim validated with [SOURCE N]
4. **Live RAGAS Evaluation** - Quality metrics on every query
5. **Full Observability** - Langfuse traces + structured logging
6. **100% Free Tier** - No paid APIs required

## Quick Start

```bash
# 1. Clone repo
git clone https://github.com/yourname/rag-research-assistant.git
cd rag-research-assistant

# 2. Setup environment
cp .env.example .env
# Edit .env with your GROQ_API_KEY (get free key at groq.com)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run app
streamlit run app.py
```

## Project Structure

```
rag-research-assistant/
├── app.py                    # Streamlit entry
├── src/                      # Core source code
│   ├── config.py            # Centralized settings
│   ├── db/                  # ChromaDB client
│   ├── ingestion/           # PDF → chunks → embeddings
│   ├── retrieval/           # Hybrid search + reranking
│   ├── generation/          # LLM + citation validation
│   ├── monitoring/          # Logging + metrics
│   └── eval/               # Live RAGAS evaluation
├── pages/                   # Streamlit UI pages
├── tests/                   # Unit tests
├── eval/                    # Benchmark evaluation
└── data/                    # Local data storage
```

## Usage

### Upload Documents
1. Go to **Upload** page
2. Select PDF file
3. Click "Process Document"

### Ask Questions
1. Go to **Chat** page
2. Type your question
3. View cited answer with sources

### Monitor Quality
Enable "Show retrieval debug" to see:
- BM25 vs Vector hit counts
- RRF fusion scores
- Cross-encoder rerank scores
- RAGAS quality metrics

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key (free tier at groq.com) |
| `GROQ_MODEL` | No | Model: gpt-oss-120b, llama-3.3-70b-versatile |
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse tracing (open source) |
| `LANGFUSE_SECRET_KEY` | No | Langfuse secret key |
| `CHROMA_HOST` | No | Leave empty for local mode |
| `LOG_LEVEL` | No | DEBUG, INFO, WARNING |

## Tech Stack

- **Frontend**: Streamlit
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (local, free)
- **LLM**: Groq GPT-OSS-120B / Llama-3.3-70B (fast, free tier)
- **Vector DB**: ChromaDB
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Evaluation**: RAGAS
- **Observability**: Langfuse + Loguru (open source)

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run evaluation
python -m eval.runner
```

## Deployment

### Streamlit Community Cloud (Free)

1. Push code to GitHub
2. Connect repo at [share.streamlit.io](https://share.streamlit.io)
3. Add secrets in Settings
4. Deploy

### Local Docker

```bash
docker-compose up --build
```

## RAGAS Metrics

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Faithfulness | > 0.70 | Answer consistent with context |
| Answer Relevancy | > 0.70 | Answer addresses question |
| Context Precision | > 0.70 | Retrieved chunks are relevant |

## Monitoring

Structured logs include:
- Query ID and text
- Retrieval latency (vector + BM25 + rerank)
- Generation latency and token usage
- Citation count
- RAGAS scores

Example:
```
RAG Pipeline [a7f3b2d1]: 1285.2ms total
Retrieval: 45.2ms | Vector: 20 | BM25: 18
Generation: 1240ms | Tokens: 856 | Citations: 3
RAGAS: faithfulness=0.85, relevancy=0.92
```

## License

MIT License - see LICENSE file

## Contributing

1. Fork the repo
2. Create feature branch
3. Run tests: `pytest`
4. Run eval: `python -m eval.runner`
5. Submit PR

## Links

- [Live Demo](https://yourapp.streamlit.app)
- [Langfuse Dashboard](https://cloud.langfuse.com)
- [Groq Console](https://console.groq.com)
- [GitHub Repo](https://github.com/yourname/rag-research-assistant)
