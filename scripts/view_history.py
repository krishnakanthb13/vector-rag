"""
View chat history from history.md.
"""

import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# --- ANSI ---
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"

HISTORY_FILE = SCRIPT_DIR / "history.md"


def main():
    parser = argparse.ArgumentParser(description="View chat history")
    parser.add_argument("--last", type=int, default=0, help="Show last N entries (0 = all)")
    parser.add_argument("--search", type=str, help="Filter entries containing text")
    parser.add_argument("--clear", action="store_true", help="Clear history log")
    args = parser.parse_args()

    if args.clear:
        if HISTORY_FILE.exists():
            confirm = input(f"{YELLOW}Clear history.md? (y/n): {RESET}").strip().lower()
            if confirm == "y":
                HISTORY_FILE.write_text("", encoding="utf-8")
                print(f"{GREEN}History cleared.{RESET}")
            else:
                print("Cancelled.")
        else:
            print("No history file found.")
        return

    if not HISTORY_FILE.exists():
        print(f"{DIM}No history.md found. Start a chat to create one.{RESET}")
        return

    content = HISTORY_FILE.read_text(encoding="utf-8")

    entries = []
    current = []
    for line in content.splitlines():
        if line.strip() == "---":
            if any(l.strip() for l in current):
                entries.append("\n".join(current))
            current = []
        current.append(line)
    if any(l.strip() for l in current):
        entries.append("\n".join(current))

    if args.search:
        terms = args.search.lower().split()
        entries = [e for e in entries if all(term in e.lower() for term in terms)]

    if args.last > 0:
        entries = entries[-args.last:]

    if not entries:
        print(f"{DIM}No matching entries found.{RESET}")
        return

    print(f"\n{BOLD}Chat History{RESET} ({len(entries)} entries)\n")
    print(f"{DIM}{'=' * 60}{RESET}")

    for entry in entries:
        lines = entry.strip().splitlines()
        for line in lines:
            if line.startswith("### ["):
                ts = line.replace("### [", "").rstrip("]")
                print(f"\n{YELLOW}[{ts}]{RESET}")
            elif line.startswith("**Q:**"):
                print(f"\n{GREEN}Q:{RESET} {line.replace('**Q:**', '').strip()}")
            elif line.startswith("> "):
                print(f"{DIM}{line.replace('> ', '')}{RESET}")
            elif line.startswith("---"):
                pass
            else:
                if len(line) > 300:
                    line = line[:300] + "..."
                if line.strip():
                    print(f"{CYAN}{line}{RESET}")
        print(f"\n{DIM}{'-' * 60}{RESET}")

    print()


if __name__ == "__main__":
    main()
