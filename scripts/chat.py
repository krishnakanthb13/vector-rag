"""
Interactive CLI chat with your documents via RAG.
Local model for embeddings, Gemini for the chat LLM.
"""

import os
import sys
import time
from datetime import datetime
from collections import deque
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

try:
    import colorama
    colorama.just_fix_windows_console()
except (ImportError, AttributeError):
    pass

# --- Config ---
CHROMA_DIR = str(ROOT_DIR / os.environ.get("CHROMA_DIR", "chroma_db"))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "my_documents")
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gemini-2.0-flash")
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "gemma-3-27b-it")
TOP_K = int(os.environ.get("TOP_K", "20"))
MIN_SIMILARITY = float(os.environ.get("MIN_SIMILARITY", "0.25"))
HISTORY_LOG = os.environ.get("HISTORY_LOG", str(Path(__file__).parent / "history.md"))

# Default system prompt — override via SYSTEM_PROMPT env var or .env file
DEFAULT_SYSTEM_PROMPT = """You are a precise technical assistant. Answer based ONLY on the provided document excerpts.

RULES:
1. Answer ONLY based on the provided excerpts. Do not guess or use outside knowledge.
2. If the excerpts do not contain enough information, say "The documents do not provide enough detail on this topic."
3. Be specific and detailed. Include step-by-step procedures when available.
4. Cite sources when possible (e.g., document name, section heading).
5. Synthesize information from multiple excerpts into a coherent answer.
6. Format your response in Markdown. Use **bold** for key terms and bullet lists for multi-item answers."""

SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

# --- ANSI ---
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"

DIVIDER = f"{DIM}{'=' * 60}{RESET}"
THIN_DIV = f"{DIM}{'-' * 60}{RESET}"


def log_to_file(timestamp, query, answer, sources_used, metrics=None):
    """Append conversation to history.md."""
    try:
        if os.path.exists(HISTORY_LOG) and os.path.getsize(HISTORY_LOG) > 10 * 1024 * 1024:
            date_str = datetime.now().strftime("%Y-%m-%d-%H%M")
            archive_log = str(Path(HISTORY_LOG).parent / f"history-{date_str}.md")
            os.rename(HISTORY_LOG, archive_log)
    except Exception as e:
        print(f"\n  {RED}Warning: Could not rotate history.md ({e}){RESET}\n")

    try:
        with open(HISTORY_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n---\n\n")
            f.write(f"### [{timestamp}]\n\n")
            f.write(f"**Q:** {query}\n\n")
            f.write(f"{answer}\n\n")
            f.write(f"> Sources: {', '.join(sources_used)}\n")
            if metrics:
                f.write(f"> Latency: {metrics.get('latency', '?')}s | Model: {metrics.get('model', '?')}\n")
                f.write(f"> Context: {metrics.get('context_chars', 0):,} chars | Top Match: {metrics.get('top_score', 0):.2f}\n")
            f.write(f"\n")
    except OSError as e:
        print(f"\n  {RED}Warning: Could not save to history.md ({e}){RESET}\n")


def print_banner():
    print()
    print(DIVIDER)
    print(f"{BOLD}{CYAN}  Document Chat (RAG){RESET}")
    print(f"  {DIM}Embeddings: {EMBED_MODEL} (local) | Chat: {CHAT_MODEL} (Gemini){RESET}")
    print(f"  {DIM}Retrieved chunks: {TOP_K}{RESET}")
    print(DIVIDER)
    print(f"  {GREEN}Ask anything{RESET} about your documents.")
    print()
    print(f"  {YELLOW}Commands:{RESET}")
    print(f"    {CYAN}/stats{RESET}    Show collection stats")
    print(f"    {CYAN}/history{RESET}  Show recent chat history")
    print(f"    {CYAN}/help{RESET}     Show this help")
    print(f"    {CYAN}/quit{RESET}     Exit")
    print()
    print(DIVIDER)
    print()


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(f"{RED}ERROR: Set GEMINI_API_KEY in your .env file.{RESET}")
        return

    print(f"{DIM}Loading embedding model ({EMBED_MODEL})...{RESET}")
    embed_model = SentenceTransformer(EMBED_MODEL)

    gemini = genai.Client(api_key=api_key)

    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma.get_collection(COLLECTION_NAME)
    print(f"{GREEN}Loaded {collection.count()} embedded chunks.{RESET}")

    print_banner()

    history = deque(maxlen=100)

    while True:
        try:
            query = input(f"{BOLD}{GREEN}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Bye!{RESET}")
            break

        if not query:
            continue

        if query in ("/quit", "/exit", "/q"):
            print(f"{DIM}Bye!{RESET}")
            break

        if query == "/stats":
            col_meta = collection.metadata or {}
            print(f"\n  {CYAN}Chunks: {collection.count()}{RESET}")
            for k, v in col_meta.items():
                print(f"  {CYAN}{k}: {v}{RESET}")
            print()
            continue

        if query == "/help":
            print_banner()
            continue

        if query == "/history":
            if not history:
                print(f"\n  {DIM}No conversation history yet.{RESET}\n")
            else:
                print(f"\n{THIN_DIV}")
                for role, msg in list(history)[-10:]:
                    if role == "user":
                        print(f"  {GREEN}You:{RESET} {msg[:80]}...")
                    else:
                        print(f"  {CYAN}Bot:{RESET} {msg[:80]}...")
                print(f"{THIN_DIV}\n")
            continue

        # --- Retrieve relevant chunks ---
        query_emb = embed_model.encode([query], normalize_embeddings=True).tolist()

        results = collection.query(
            query_embeddings=query_emb,
            n_results=TOP_K * 3,
        )

        seen = set()
        filtered_docs = []
        filtered_meta = []
        filtered_scores = []

        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            score = 1 - results["distances"][0][i]

            if score < MIN_SIMILARITY:
                continue

            key = (meta.get("heading", ""), meta.get("source_file", ""))
            if key not in seen:
                seen.add(key)
                filtered_docs.append(doc)
                filtered_meta.append(meta)
                filtered_scores.append(score)
                if len(filtered_docs) >= TOP_K:
                    break

        if not filtered_docs:
            print(f"\n  {RED}No relevant results found (below threshold).{RESET}\n")
            continue

        # --- Build context ---
        MAX_CONTEXT_CHARS = 100000
        context_parts = []
        sources_used = []
        for i, doc in enumerate(filtered_docs):
            meta = filtered_meta[i]
            heading = meta.get("heading", "")
            source = meta.get("source_file", "")
            score = filtered_scores[i]
            heading_tag = f" [{heading}]" if heading else ""
            source_tag = f" ({source})" if source else ""

            part = f"---{heading_tag}{source_tag} ---\n{doc}"

            current_len = sum(len(p) for p in context_parts)
            if current_len + len(part) > MAX_CONTEXT_CHARS:
                if not context_parts:
                    part = part[:MAX_CONTEXT_CHARS] + "\n...[truncated]"
                    context_parts.append(part)
                    sources_used.append(f"{source or 'unknown'} ({score:.2f})")
                break

            context_parts.append(part)
            sources_used.append(f"{source or 'unknown'}:{heading or '?'} ({score:.2f})")

        context = "\n\n".join(context_parts)

        # --- Build prompt ---
        user_prompt = f"Document excerpts:\n\n{context}\n\n---\nQuestion: {query}"

        contents = []
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=SYSTEM_PROMPT)],
        ))
        contents.append(types.Content(
            role="model",
            parts=[types.Part.from_text(text="Understood.")],
        ))
        for role, msg in list(history)[-5:]:
            contents.append(types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg)],
            ))
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_prompt)],
        ))

        metrics = {
            "top_score": filtered_scores[0] if filtered_scores else 0.0,
            "context_chars": sum(len(p) for p in context_parts)
        }

        # --- Call LLM (primary) then fallback ---
        print(f"\n  {DIM}Searching...{RESET}", end="", flush=True)
        answer = None
        t0 = time.time()
        for model in [CHAT_MODEL, FALLBACK_MODEL]:
            try:
                response = gemini.models.generate_content(
                    model=model,
                    contents=contents,
                )
                answer = response.text
                break
            except Exception as e:
                err_str = str(e)
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    answer = f"{RED}Rate limited by API. Please wait and retry.{RESET}"
                    break
                if model == CHAT_MODEL:
                    print(f"\r  {YELLOW}Primary model busy, trying fallback...{RESET}  ", end="", flush=True)
                else:
                    answer = f"{RED}Error: Both models unavailable. {e}{RESET}"
        print(f"\r", end="")

        metrics["latency"] = round(time.time() - t0, 2)
        metrics["model"] = model

        # --- Display ---
        print(f"\n{BOLD}{CYAN}Assistant:{RESET}")
        print(f"{answer}")
        print(f"  {DIM}Context: {metrics['context_chars']:,} chars | Top Match: {metrics['top_score']:.2f}{RESET}")
        print(f"  {DIM}Sources: {', '.join(sources_used)}{RESET}")
        print()

        # --- Log ---
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_to_file(timestamp, query, answer, sources_used, metrics)

        history.append(("user", query))
        history.append(("model", answer))


if __name__ == "__main__":
    main()
