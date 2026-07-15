"""
Query the ChromaDB vector store.
Uses local sentence-transformers model for query embeddings.
"""

import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

import os
from sentence_transformers import SentenceTransformer
import chromadb

try:
    import colorama
    colorama.just_fix_windows_console()
except (ImportError, AttributeError):
    pass

# --- Config ---
CHROMA_DIR = str(ROOT_DIR / os.environ.get("CHROMA_DIR", "chroma_db"))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "my_documents")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_TOP_K = int(os.environ.get("TOP_K", "20"))

# --- ANSI ---
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser(
        description="Search vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""{CYAN}Examples:{RESET}
  python query_embeddings.py "how to configure"
  python query_embeddings.py "settings" --top 10
  python query_embeddings.py --list --top 20
        """,
    )
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_K,
                        help=f"Number of results (default: {DEFAULT_TOP_K})")
    parser.add_argument("--list", action="store_true",
                        help="List stored documents")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Minimum similarity score (0-1, default: 0)")
    args = parser.parse_args()

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    print(f"{BOLD}Collection:{RESET} {COLLECTION_NAME} ({collection.count()} chunks)\n")

    # --- List mode ---
    if args.list:
        get_params = {"limit": args.top}
        results = collection.get(**get_params)
        print(f"{CYAN}{'#':<6} {'Heading':<30} Preview{RESET}")
        print(f"{'-' * 80}")
        for i, doc_id in enumerate(results["ids"]):
            meta = results["metadatas"][i]
            heading = meta.get("heading", "")[:28]
            preview = results["documents"][i][:50].replace("\n", " ")
            print(f"{i+1:<6} {heading:<30} {preview}...")
        return

    # --- Semantic search ---
    if not args.query:
        parser.print_help()
        return

    print(f"{DIM}Loading model...{RESET}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    search_text = args.query
    query_embedding = model.encode([search_text], normalize_embeddings=True).tolist()

    query_params = {"query_embeddings": query_embedding, "n_results": args.top}
    results = collection.query(**query_params)

    filtered = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        score = 1 - distance
        if score >= args.min_score:
            filtered.append({
                "id": results["ids"][0][i],
                "doc": results["documents"][0][i],
                "meta": results["metadatas"][0][i],
                "score": score,
            })
        if len(filtered) >= args.top:
            break

    if not filtered:
        print(f"{RED}No results found.{RESET}")
        return

    print(f"\n{BOLD}Query:{RESET} \"{args.query}\"\n")

    for i, r in enumerate(filtered):
        heading = r["meta"].get("heading", "")
        source = r["meta"].get("source_file", "")
        preview = r["doc"][:200].replace("\n", " ")

        if r["score"] >= 0.5:
            score_color = GREEN
        elif r["score"] >= 0.3:
            score_color = YELLOW
        else:
            score_color = RED

        heading_tag = f" {YELLOW}{heading}{RESET}" if heading else ""
        source_tag = f" {DIM}({source}){RESET}" if source else ""
        print(f"  {BOLD}[{i+1}]{RESET}{heading_tag}{source_tag} "
              f"(score: {score_color}{r['score']:.3f}{RESET})")
        print(f"      {DIM}{preview}...{RESET}")
        print()


if __name__ == "__main__":
    main()
