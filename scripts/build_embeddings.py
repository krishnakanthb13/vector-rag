"""
Build ChromaDB vector store from .md files using a local embedding model.
Processes each file, splits at ## headings, and embeds chunks.
"""

import os
import re
import glob
import time
import hashlib
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    import colorama
    colorama.just_fix_windows_console()
except (ImportError, AttributeError):
    pass

# --- Config (all from .env) ---
SOURCE_DIR = Path(os.environ.get("SOURCE_DIR", str(ROOT_DIR / "source_docs")))
CHROMA_DIR = ROOT_DIR / os.environ.get("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "my_documents")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
BATCH_SIZE = 64
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 2000

# --- ANSI ---
RED = "\033[31m"
RESET = "\033[0m"


def strip_frontmatter(text: str) -> tuple[str, dict]:
    """Remove YAML front-matter and image refs, return clean text + metadata."""
    metadata = {}
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if fm_match:
        for line in fm_match.group(1).strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                metadata[key.strip()] = val.strip().strip('"').strip("'")
        text = text[fm_match.end():]
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, metadata


def chunk_by_headings(text: str, source_file: str) -> list[dict]:
    """Split text at ## headings, respecting max chunk size."""
    chunks = []
    sections = re.split(r"(?=^##\s)", text, flags=re.MULTILINE)

    for i, section in enumerate(sections):
        section = section.strip()
        if not section or len(section) < MIN_CHUNK_SIZE:
            continue

        heading_match = re.match(r"^##+\s+(.+)", section)
        heading = heading_match.group(1).strip() if heading_match else f"Section {i+1}"

        if len(section) > MAX_CHUNK_SIZE:
            paragraphs = section.split("\n\n")
            current = ""
            for para in paragraphs:
                sub_paras = []
                if len(para) > MAX_CHUNK_SIZE:
                    curr_p = para
                    while len(curr_p) > MAX_CHUNK_SIZE:
                        split_idx = max(curr_p.rfind(" ", 0, MAX_CHUNK_SIZE),
                                        curr_p.rfind("\n", 0, MAX_CHUNK_SIZE))
                        if split_idx == -1:
                            split_idx = MAX_CHUNK_SIZE
                        sub_paras.append(curr_p[:split_idx].strip())
                        curr_p = curr_p[split_idx:].lstrip()
                    if curr_p:
                        sub_paras.append(curr_p.strip())
                else:
                    sub_paras = [para]

                for sp in sub_paras:
                    if len(current) + len(sp) > MAX_CHUNK_SIZE and current:
                        chunks.append({
                            "text": current.strip(),
                            "heading": heading,
                            "chunk_index": len(chunks),
                            "source_file": source_file,
                        })
                        current = sp
                    else:
                        current = current + "\n\n" + sp if current else sp
            if current.strip() and len(current.strip()) >= MIN_CHUNK_SIZE:
                chunks.append({
                    "text": current.strip(),
                    "heading": heading,
                    "chunk_index": len(chunks),
                    "source_file": source_file,
                })
        else:
            chunks.append({
                "text": section,
                "heading": heading,
                "chunk_index": i,
                "source_file": source_file,
            })

    return chunks


def collect_files() -> list[Path]:
    files = sorted(glob.glob(str(SOURCE_DIR / "*.md")))
    print(f"Found {len(files)} .md files in {SOURCE_DIR}/")
    return [Path(f) for f in files]


def main():
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    CHROMA_DIR.mkdir(exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    TEMP_COLLECTION_NAME = f"{COLLECTION_NAME}_temp"
    try:
        chroma_client.delete_collection(TEMP_COLLECTION_NAME)
        print(f"Deleted existing temp collection '{TEMP_COLLECTION_NAME}'")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=TEMP_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    files = collect_files()
    documents, metadatas, ids = [], [], []
    seen_ids = set()
    duplicate_count = 0

    for fpath in tqdm(files, desc="Chunking files"):
        raw = fpath.read_text(encoding="utf-8", errors="replace")
        clean_text, meta = strip_frontmatter(raw)
        if not clean_text:
            continue

        chunks = chunk_by_headings(clean_text, fpath.name)

        for chunk in chunks:
            content_str = f"{chunk['heading']}_{chunk['text']}".encode('utf-8')
            doc_id = hashlib.sha1(content_str).hexdigest()
            if doc_id in seen_ids:
                duplicate_count += 1
                continue
            seen_ids.add(doc_id)
            chunk_meta = {
                "source_file": chunk["source_file"],
                "heading": chunk["heading"],
                "chunk_index": chunk["chunk_index"],
            }
            documents.append(chunk["text"])
            metadatas.append(chunk_meta)
            ids.append(doc_id)

    print(f"\nProcessed {len(documents)} chunks from {len(files)} files")

    print(f"Embedding locally via {EMBEDDING_MODEL_NAME} (batch size: {BATCH_SIZE})...")
    t0 = time.time()

    for start in tqdm(range(0, len(documents), BATCH_SIZE), desc="Embedding"):
        batch_docs = documents[start : start + BATCH_SIZE]
        batch_meta = metadatas[start : start + BATCH_SIZE]
        batch_ids = ids[start : start + BATCH_SIZE]

        embeddings = model.encode(batch_docs, show_progress_bar=False, normalize_embeddings=True).tolist()

        collection.add(
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=batch_meta,
            ids=batch_ids,
        )

    elapsed = time.time() - t0

    collection.modify(metadata={
        "total_chunks": len(documents),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "build_timestamp": datetime.now().isoformat()
    })

    # Swap collections
    if collection.count() == 0:
        print(f"\n{RED}Error: New collection is empty. Aborting swap.{RESET}")
        return

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"Deleted old collection '{COLLECTION_NAME}'")
    except Exception:
        pass
    collection.modify(name=COLLECTION_NAME)

    print(f"\nDone! Embedded {collection.count()} chunks in {elapsed:.1f}s")
    if duplicate_count > 0:
        print(f"Skipped {duplicate_count} duplicate chunks.")
    print(f"Collection: {COLLECTION_NAME} in {CHROMA_DIR}/")
    print(f"Test with: python scripts/query_embeddings.py \"your query\"")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\033[31mProcess interrupted by user (Ctrl+C). Cleaning up...\033[0m")
        try:
            import chromadb
            chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            chroma_client.delete_collection(f"{COLLECTION_NAME}_temp")
            print(f"\033[2mCleaned up temporary collection '{COLLECTION_NAME}_temp'.\033[0m")
        except Exception:
            pass
        sys.exit(1)
