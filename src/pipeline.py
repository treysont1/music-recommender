"""
Recommendation pipeline orchestrator.

Coordinates:
  validate → extract_prefs → parallel retrieve → fuse → generate
"""

from rag import retrieve, generate, K_RETRIEVE
from ranking import extract_preferences, fuse, KEYWORD_BANK
from recommender import load_songs, recommend_songs
from pathlib import Path

# Constants
K_FINAL = 10
LOW_CONFIDENCE_THRESHOLD = 0.3
MIN_QUERY_LENGTH = 3
EXTRA_MUSIC_WORDS = {
    "song", "music", "listen", "playlist", "vibe", "beat", "track",
    "artist", "genre", "mood", "tempo", "energy", "relax", "morning",
    "night", "jazz", "indie",
}
MUSIC_KEYWORDS = set(KEYWORD_BANK.keys()) | EXTRA_MUSIC_WORDS

ROOT = Path(__file__).parent.parent
TRACKS_CSV = ROOT / "data" / "spotify_tracks.csv"
_all_songs = load_songs(str(TRACKS_CSV))


def _validate_query(query: str) -> str | None:
    """
    Returns an error message if the query should be rejected, otherwise None.
    Checks for minimum length and whether the query seems music-related.
    """
    if len(query.strip()) < MIN_QUERY_LENGTH:
        return "Query is too short. Please describe what you want to listen to."

    words = set(query.lower().split())
    if not words & MUSIC_KEYWORDS:
        return (
            "Your query doesn't seem music-related. "
            "Try describing a mood, activity, genre, or vibe (e.g. 'chill music for studying')."
        )

    return None


def _retrieve_rules(prefs: dict, k: int) -> list[dict]:
    if not prefs:
        return []
    scored = recommend_songs(prefs, _all_songs, k=k)
    return [song for song, _score, _explanation in scored]


def recommend(query: str) -> str:
    """
    Takes a natural language query and returns a Gemini-generated recommendation.
    Requires build_index() to have been run first.
    """
    error = _validate_query(query)
    if error:
        return f"[Input validation] {error}"

    prefs = extract_preferences(query)
    
    rag_songs, similarities = retrieve(query, k=K_RETRIEVE)
    rule_songs = _retrieve_rules(prefs, k=K_RETRIEVE)

    avg_similarity = round(sum(similarities) / len(similarities), 3)
    print(f"Retrieval confidence — top {K_RETRIEVE} RAG songs, avg similarity: {avg_similarity} "
        f"(range: {min(similarities)}–{max(similarities)})\n")


    if avg_similarity < LOW_CONFIDENCE_THRESHOLD:
        return (
            f"[Low confidence — avg similarity: {avg_similarity}] "
            "The catalog doesn't have strong matches for that query. "
            "Try rephrasing with a genre, mood, or activity."
        )
    
    fused_songs = fuse(rag_songs, rule_songs, k_final=K_FINAL)
    return generate(query, fused_songs)
