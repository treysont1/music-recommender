"""
RAG module: embeds the song catalog locally with sentence-transformers,
stores vectors in ChromaDB, and generates recommendations with Gemini.

Build the index once:
    from rag import build_index
    build_index()

Then query:
    from rag import recommend
    print(recommend("something chill to study to on a rainy day"))
"""

import os
import csv
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
import chromadb

load_dotenv()

gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
embedder = SentenceTransformer("all-MiniLM-L6-v2")

ROOT = Path(__file__).parent.parent
TRACKS_CSV = ROOT / "data" / "spotify_tracks.csv"
CHROMA_DIR = str(ROOT / "data" / "chroma_db")
COLLECTION_NAME = "spotify_tracks"
GEN_MODEL = "gemini-2.5-flash"
K_RETRIEVE = 50

def _song_to_text(song: dict) -> str:
    return (
        f"{song['title']} by {song['artist']}. "
        f"Genre: {song['genre']}. Mood: {song['mood']}. "
        f"Energy: {song['energy']}, Valence: {song['valence']}, "
        f"Tempo: {song['tempo_bpm']} BPM, Danceability: {song['danceability']}, "
        f"Acousticness: {song['acousticness']}."
    )


def _load_tracks() -> list[dict]:
    with open(TRACKS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _get_collection(chroma_client: chromadb.PersistentClient):
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_index() -> None:
    """Embeds all tracks locally and stores them in ChromaDB. Safe to re-run."""
    tracks = _load_tracks()
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = _get_collection(chroma_client)

    existing = set(collection.get()["ids"])
    to_index = [t for t in tracks if t["title"] + "|" + t["artist"] not in existing]

    if not to_index:
        print("Index already up to date.")
        return

    print(f"Indexing {len(to_index)} tracks...")

    texts = [_song_to_text(t) for t in to_index]
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True)

    collection.add(
        ids=[t["title"] + "|" + t["artist"] for t in to_index],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{k: v for k, v in t.items()} for t in to_index],
    )

    print("Index complete.")


def retrieve(query: str, k: int = K_RETRIEVE) -> tuple[list[dict], list[float]]:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = _get_collection(chroma_client)

    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=k, include=["metadatas", "distances"])

    # ChromaDB returns cosine distances (0=identical, 2=opposite).
    # Convert to similarity scores in the 0.0–1.0 range.
    similarities = [round(1 - (d / 2), 3) for d in results["distances"][0]]
    return results["metadatas"][0], similarities

def generate(query: str, songs: list[dict]) -> str:
    """
    Formats the retrieved songs into a prompt and asks Gemini to recommend
    the top 5 with explanations.
    """
    song_list = "\n".join(
        f"{i+1}. \"{s['title']}\" by {s['artist']} "
        f"(genre: {s['genre']}, mood: {s['mood']}, energy: {s['energy']}, "
        f"valence: {s['valence']})"
        for i, s in enumerate(songs)
    )

    prompt = f"""You are a music recommendation assistant.

A user is looking for: "{query}"

Based on their request, here are the most relevant songs retrieved from the catalog:

{song_list}

Recommend the top 5 songs from this list that best match the user's request.
For each song, explain in one sentence why it fits. Be conversational and specific."""

    response = gemini_client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            thinking_config=genai.types.ThinkingConfig(thinking_budget=0)
        ),
    )
    return response.text
