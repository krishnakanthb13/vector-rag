# Vector RAG Toolkit

A generic, reusable **Retrieval-Augmented Generation** pipeline. Point it at any folder of `.md` files and get a local vector store + interactive chat.

## Quick Start

### 1. Install dependencies

```bash
pip install chromadb sentence-transformers google-genai python-dotenv tqdm colorama
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set your GEMINI_API_KEY and adjust COLLECTION_NAME
```

### 3. Add your documents

Place `.md` files in `source_docs/`. Each file should have `##` headings for best chunking.

### 4. Build embeddings

```bash
python scripts/build_embeddings.py
```

Or double-click `rag.bat` and choose option 1.

### 5. Chat

```bash
python scripts/chat.py
```

Or use `rag.bat` option 3 for an interactive menu.

## Project Structure

```
vector-rag/
├── .env.example        # Config template
├── .env                # Your config (git-ignored)
├── rag.bat             # Windows menu launcher
├── README.md           # This file
├── SKILL.md            # Build-your-own RAG guide
├── scripts/
│   ├── build_embeddings.py   # Chunk + embed → ChromaDB
│   ├── query_embeddings.py   # Search the vector store
│   ├── chat.py               # Interactive RAG chat
│   └── view_history.py       # Browse chat history
├── source_docs/        # Put your .md files here
├── chroma_db/          # Vector store (auto-created)
└── notes/
    └── about-rag.md    # RAG concepts reference
```

## Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_DIR` | `./chroma_db` | Where the vector store lives |
| `COLLECTION_NAME` | `my_documents` | Name of your collection |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers model |
| `CHAT_MODEL` | `gemini-2.0-flash` | Primary LLM |
| `FALLBACK_MODEL` | `gemma-3-27b-it` | Fallback if primary is rate-limited |
| `GEMINI_API_KEY` | — | Your Google AI API key |
| `SOURCE_DIR` | `./source_docs` | Folder with `.md` files |
| `TOP_K` | `20` | Chunks retrieved per query |
| `MIN_SIMILARITY` | `0.25` | Minimum score threshold (0–1) |

## Scripts

| Script | Purpose |
|--------|---------|
| `build_embeddings.py` | Reads `source_docs/*.md`, chunks by `##` headings, embeds, stores in ChromaDB |
| `query_embeddings.py` | CLI search — semantic search, list documents, filter by score |
| `chat.py` | Interactive chat — retrieves context, sends to Gemini, logs history |
| `view_history.py` | Browse/search/clear chat logs |

## Customization

- **Different embedding model**: Change `EMBEDDING_MODEL` in `.env` (any sentence-transformers model)
- **Different LLM**: Change `CHAT_MODEL`/`FALLBACK_MODEL` (any Gemini model)
- **Different vector store**: Replace ChromaDB calls in scripts with Pinecone/Weaviate/Qdrant
- **Different chunking**: Edit `chunk_by_headings()` in `build_embeddings.py`
- **Custom system prompt**: Set `SYSTEM_PROMPT` in `.env` or edit the default in `chat.py`

## Requirements

- Python 3.10+
- Google Gemini API key (free tier available)
- ~500MB disk for embedding model (first run downloads it)
