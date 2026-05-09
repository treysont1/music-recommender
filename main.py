import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag import build_index
from pipeline import recommend


def main() -> None:
    print("Music Recommender — powered by Gemini + ChromaDB")
    print("Building index (skipped if already up to date)...")
    build_index()

    print("\nDescribe what you want to listen to. Type 'quit' to exit.\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break

        print("\nFinding recommendations...\n")
        result = recommend(query)
        print(result)
        print()


if __name__ == "__main__":
    main()
