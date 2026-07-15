# Vector RAG Notes

## What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that enhances Large Language Models (LLMs) by combining them with external knowledge retrieval. Instead of relying solely on the model's training data, RAG retrieves relevant documents from a knowledge base at query time and injects them as context into the LLM prompt.

## Core Components

### 1. Document Store (Source Material)
Your raw documents — PDFs, Markdown files, text files, etc. These need to be converted to a text format the pipeline can process.

### 2. Text Chunking
Documents are split into smaller pieces (chunks) for embedding. Common strategies:
- **Fixed-size**: Split every N characters/tokens
- **Heading-based**: Split at markdown headings (`##`, `###`)
- **Semantic**: Use an LLM to find natural boundaries
- **Recursive**: Try heading, then paragraph, then sentence splits

### 3. Embedding Model
Converts text chunks into dense vector representations (embeddings). Popular options:
- **all-MiniLM-L6-v2** — fast, local, 384 dimensions (default in this toolkit)
- **text-embedding-3-small** — OpenAI, 1536 dimensions
- **bge-large-en-v1.5** — high quality, local, 1024 dimensions
- **voyage-3** — Voyage AI, high quality

### 4. Vector Database
Stores embeddings and enables fast similarity search:
- **ChromaDB** — lightweight, local, easy setup (used in this toolkit)
- **Pinecone** — managed, scalable
- **Weaviate** — open-source, feature-rich
- **Qdrant** — high performance, Rust-based
- **FAISS** — Meta's library, very fast, low-level

### 5. Retrieval
When a user queries, the query is embedded and the top-k most similar chunks are retrieved from the vector store. Key parameters:
- **TOP_K**: How many chunks to retrieve (more context but higher cost)
- **MIN_SIMILARITY**: Threshold to filter irrelevant results
- **Metadata filters**: Narrow by page, date, category, etc.

### 6. LLM (Generation)
Takes the retrieved context + user query and generates an answer. Options:
- **Gemini** (Google) — fast, affordable
- **GPT-4o** (OpenAI) — strong reasoning
- **Claude** (Anthropic) — nuanced, careful
- **Local LLMs** (Llama, Mistral) — free, runs on your machine

## The RAG Flow

```
Documents → Chunk → Embed → Store in Vector DB
                                    ↓
User Query → Embed → Similarity Search → Top-K Chunks
                                              ↓
                                    Inject into LLM Prompt
                                              ↓
                                        Answer
```

## Common Pitfalls

1. **Chunks too large** — reduces retrieval precision, wastes context window
2. **Chunks too small** — loses context, creates fragmented answers
3. **No deduplication** — same content appears multiple times, biases results
4. **Wrong embedding model** — domain mismatch reduces relevance
5. **No minimum similarity** — irrelevant chunks pollute the context
6. **Context too long** — exceeds LLM context window, truncation loses info

## References

- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)
- LangChain RAG documentation
- ChromaDB documentation
- sentence-transformers documentation
