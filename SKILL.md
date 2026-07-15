# SKILL.md — Build Your Own RAG Pipeline

A step-by-step guide to building a Retrieval-Augmented Generation pipeline from scratch. Pick your components at each stage and wire them together.

## Overview

RAG = **Retrieve** relevant documents → **Augment** the prompt → **Generate** an answer.

```
Documents → Chunk → Embed → Store
                              ↓
Query → Embed → Search → Top-K → LLM → Answer
```

## Step 1: Prepare Your Documents

**Goal**: Get your source material into text format.

| Source | Tool | Command |
|--------|------|---------|
| PDF | pymupdf4llm | `pip install pymupdf4llm` → `pymupdf4llm.to_markdown("file.pdf")` |
| Word (.docx) | python-docx | `pip install python-docx` |
| Web pages | requests + BeautifulSoup | `pip install requests beautifulsoup4` |
| Plain text | Built-in | `open("file.txt").read()` |
| Images (OCR) | Tesseract | `pip install pytesseract` |

**Output**: A folder of `.md` or `.txt` files, one per document or section.

**Tip**: Use `##` headings in your markdown — the default chunker splits on them.

## Step 2: Choose a Text Chunking Strategy

**Goal**: Split documents into pieces small enough for embedding but large enough for context.

| Strategy | Best For | Implementation |
|----------|----------|----------------|
| **Heading-based** (default) | Structured docs with clear sections | Split on `## ` regex |
| **Fixed-size** | Uniform content (logs, transcripts) | Split every N characters |
| **Recursive character** | General purpose | Try `\n\n`, then `\n`, then ` ` |
| **Semantic** | When boundaries matter | Use an LLM to find natural breaks |

**Key parameters**:
- `MIN_CHUNK_SIZE = 100` — skip chunks smaller than this
- `MAX_CHUNK_SIZE = 2000` — split oversized chunks further

**Deduplication**: Hash each chunk's content to skip duplicates.

## Step 3: Choose an Embedding Model

**Goal**: Convert text chunks into dense vectors for similarity search.

| Model | Dimensions | Speed | Quality | Runs Locally? |
|-------|-----------|-------|---------|---------------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good | Yes |
| `all-mpnet-base-v2` | 768 | Medium | Better | Yes |
| `bge-large-en-v1.5` | 1024 | Slow | Great | Yes |
| `text-embedding-3-small` (OpenAI) | 1536 | API | Great | No |
| `voyage-3` (Voyage) | 1024 | API | Great | No |

**Install local models**:
```bash
pip install sentence-transformers
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(["your text here"], normalize_embeddings=True)
```

**Install OpenAI**:
```bash
pip install openai
client = OpenAI()
resp = client.embeddings.create(input=["your text"], model="text-embedding-3-small")
embedding = resp.data[0].embedding
```

## Step 4: Choose a Vector Database

**Goal**: Store embeddings and enable fast similarity search.

| Database | Type | Setup | Best For |
|----------|------|-------|----------|
| **ChromaDB** | Local, embedded | `pip install chromadb` | Quick start, small projects |
| **FAISS** | Library (Meta) | `pip install faiss-cpu` | Maximum speed, in-memory |
| **Pinecone** | Managed cloud | `pip install pinecone-client` | Production, scalability |
| **Weaviate** | Self-hosted/cloud | `pip install weaviate-client` | Rich filtering, hybrid search |
| **Qdrant** | Self-hosted/cloud | `pip install qdrant-client` | High performance |

**ChromaDB example**:
```python
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection("my_docs", metadata={"hnsw:space": "cosine"})
collection.add(documents=texts, embeddings=embeddings, metadatas=metas, ids=ids)
results = collection.query(query_embeddings=query_emb, n_results=10)
```

## Step 5: Configure Retrieval

**Goal**: Set how many chunks to retrieve and filter irrelevant ones.

| Parameter | What It Does | Suggested Value |
|-----------|-------------|-----------------|
| `TOP_K` | Number of chunks retrieved | 10–30 |
| `MIN_SIMILARITY` | Filter out low-score results | 0.2–0.4 |
| Metadata filters | Narrow by category, date, etc. | Use ChromaDB `where` clauses |

**Score interpretation** (cosine similarity):
- `> 0.5` — Strong match
- `0.3–0.5` — Moderate match
- `< 0.3` — Weak / likely irrelevant

## Step 6: Choose an LLM

**Goal**: Generate answers using retrieved context.

| Provider | Model | Free Tier? | Notes |
|----------|-------|-----------|-------|
| Google | gemini-2.0-flash | Yes | Fast, affordable |
| OpenAI | gpt-4o | No | Strong reasoning |
| Anthropic | claude-3.5-sonnet | No | Nuanced, careful |
| Local | llama-3.1-8b | Yes | Ollama + llama.cpp |
| Local | mistral-7b | Yes | Ollama + llama.cpp |

**Gemini example**:
```python
from google import genai
client = genai.Client(api_key="your-key")
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[system_prompt, context, user_query],
)
print(response.text)
```

## Step 7: Wire It Together

The pattern for all RAG scripts:

```python
# 1. Load models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_collection("my_docs")

# 2. Embed the query
query_emb = embed_model.encode([query], normalize_embeddings=True).tolist()

# 3. Retrieve
results = collection.query(query_embeddings=query_emb, n_results=20)

# 4. Build context
context = "\n\n".join(results["documents"][0])

# 5. Generate
response = llm.generate(
    system="Answer based on the following context:\n" + context,
    user=query,
)
```

## Step 8: Evaluate and Iterate

Check these metrics:
- **Top match score** — is the most relevant chunk scoring > 0.5?
- **Context coverage** — does the context contain the answer?
- **Answer accuracy** — does the LLM's answer match the source?
- **Latency** — is response time acceptable?

Common fixes:
- Chunks too large → reduce `MAX_CHUNK_SIZE`
- Chunks too small → increase `MIN_CHUNK_SIZE`
- Low relevance → try a better embedding model
- Too much noise → increase `MIN_SIMILARITY`
- Missing context → increase `TOP_K`

## Advanced Patterns

### Hybrid Search
Combine semantic (vector) + keyword (BM25) search for better recall.

### Re-ranking
Retrieve 50 chunks, then use a cross-encoder to re-rank the top 10.

### Query Expansion
Use an LLM to expand the user's query into multiple sub-queries, then merge results.

### Multi-vector Retrieval
Store both chunk summaries and full text; search summaries for broad queries, full text for specific ones.

### Agentic RAG
Let the LLM decide when to retrieve, what to search for, and how many rounds of retrieval to run.

## Resources

- [LangChain RAG docs](https://python.langchain.com/docs/tutorials/rag/)
- [ChromaDB docs](https://docs.trychroma.com/)
- [sentence-transformers docs](https://www.sbert.net/)
- [Gemini API docs](https://ai.google.dev/gemini-api/docs)
