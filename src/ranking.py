import re
import collections

KEYWORD_BANK = {
    "chill":      {"mood": "chill",   "valence": 0.35, "energy": 0.3},
    "calm":       {"mood": "chill",   "energy": 0.2},
    "mellow":     {"mood": "chill",   "valence": 0.4,  "energy": 0.3},
    "happy":      {"mood": "happy",   "valence": 0.8},
    "upbeat":     {"mood": "happy",   "valence": 0.8,  "energy": 0.7},
    "cheerful":   {"mood": "happy",   "valence": 0.85},
    "sad":        {"mood": "sad",     "valence": 0.2},
    "melancholy": {"mood": "sad",     "valence": 0.15},
    "intense":    {"mood": "intense", "energy": 0.85},
    "hype":       {"mood": "intense", "energy": 0.9},
    "aggressive": {"mood": "intense", "energy": 0.9,   "valence": 0.3},


    "pop":        {"genre": "pop"},
    "rock":       {"genre": "rock"},
    "hop":        {"genre": "hip-hop"},      # catches "hip hop" after tokenization
    "rap":        {"genre": "hip-hop"},      # no "rap" genre exists; map to hip-hop
    "latin":      {"genre": "latin"},
    "electronic": {"genre": "electronic"},
    "edm":        {"genre": "electronic"},
    "ambient":    {"genre": "ambient",   "energy": 0.25},
    "punk":       {"genre": "punk",      "energy": 0.85},
    "blues":      {"genre": "blues"},
    "folk":       {"genre": "folk",      "acousticness": 0.7},
    "metal":      {"genre": "metal",     "energy": 0.9},
    "classical":  {"genre": "classical", "acousticness": 0.85},
    "country":    {"genre": "country"},
    "lofi":       {"genre": "lofi",      "energy": 0.3},


    "sleep":      {"energy": 0.15},
    "study":      {"energy": 0.35, "acousticness": 0.5},
    "focus":      {"energy": 0.4},
    "workout":    {"energy": 0.85, "valence": 0.7},
    "gym":        {"energy": 0.85},
    "party":      {"energy": 0.85, "valence": 0.8, "danceability": 0.85},
    "drive":      {"energy": 0.6},


    "slow":       {"tempo_bpm": 75},
    "fast":       {"tempo_bpm": 140},


    "acoustic":   {"acousticness": 0.85},
    "unplugged":  {"acousticness": 0.9},


    "dance":      {"danceability": 0.85},
    "club":       {"danceability": 0.85, "energy": 0.85},
}

CATEGORICAL = {"mood", "genre"}
RRF_K = 60

# song_genre: str, 
# song_mood: str, 
# song_energy: float,
# song_acousticness: float, 
# song_valence: float,
# song_tempo_bpm:

def extract_preferences(query: str) -> dict:
    """
    Maps freetext query → sparse prefs dict (only fills dimensions
    actually mentioned). Returns {} when no keywords match.

    Conflict handling:
      - Numeric dimensions: average values.
      - Categorical dimensions: keep if all proposals agree, drop otherwise.

    Negation is not yet supported.
    """
    by_dimension = collections.defaultdict(list)
    result = {}

    tokens = set(re.findall(r"\b\w+\b", query.lower()))
    
    for token in tokens:
        if token not in KEYWORD_BANK:
            continue
        for dimension, value in KEYWORD_BANK[token].items():
            by_dimension[dimension].append(value)

    for dimension, values in by_dimension.items():
        if dimension in CATEGORICAL:
            if len(set(values)) == 1:
                result[dimension] = values[0]
        else:
            value = round(sum(values) / len(values), 3)
            result[dimension] = value
    
    return result



def fuse(rag_songs: list[dict], rule_songs: list[dict], k_final: int = 10) -> list[dict]:
    """
    Fuses two ranked lists of songs using Reciprocal Rank Fusion.

    Args:
        rag_songs: songs ranked by RAG similarity (top first).
        rule_songs: songs ranked by rule-based score (top first).
        k_final: number of fused results to return.

    Returns:
        Top k_final songs in fused-rank order.
    """
    scores = collections.defaultdict(float)
    song_lookup = {}

    for index, song in enumerate(rag_songs):
        rank = index + 1
        song_id = song["title"] + "|" + song["artist"]
        scores[song_id] += 1 / (RRF_K + rank)
        song_lookup[song_id] = song

    for index, song in enumerate(rule_songs):
        rank = index + 1
        song_id = song["title"] + "|" + song["artist"]
        scores[song_id] += 1 / (RRF_K + rank)
        song_lookup[song_id] = song

    sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    top = [song_lookup[sid] for sid, _score in sorted_items[:k_final]]

    return top



if __name__ == "__main__":
    # ---- extract_preferences sanity (existing) ----
    print("=== extract_preferences ===")
    for q in ["chill workout", "happy upbeat", "chill intense", "hello there", ""]:
        print(f"{q!r:25s} → {extract_preferences(q)}")

    # ---- fuse() sanity ----
    print("\n=== fuse() ===")

    def _song(title, artist="X"):
        return {"title": title, "artist": artist}

    A = _song("A")
    B = _song("B")
    C = _song("C")
    D = _song("D")
    E = _song("E")

    # Case 1: overlap — songs in both lists should outrank songs in only one
    print("\nCase 1: overlap (rag=[A,B,C], rule=[B,C,D])")
    result = fuse([A, B, C], [B, C, D], k_final=4)
    print(f"  result:   {[s['title'] for s in result]}")
    print(f"  expected: ['B', 'C', 'A', 'D']")
    print("    B (rank 2 in rag, rank 1 in rule):  1/62 + 1/61 = 0.0325")
    print("    C (rank 3 in rag, rank 2 in rule):  1/63 + 1/62 = 0.0320")
    print("    A (rank 1 in rag only):             1/61        = 0.0164")
    print("    D (rank 3 in rule only):            1/63        = 0.0159")

    # Case 2: empty rule list → RAG ordering preserved
    print("\nCase 2: empty rule (rag=[A,B,C], rule=[])")
    result = fuse([A, B, C], [], k_final=3)
    print(f"  result:   {[s['title'] for s in result]}")
    print(f"  expected: ['A', 'B', 'C']  (RAG ordering — sparse-prefs degradation case)")

    # Case 3: empty rag list → rule ordering preserved
    print("\nCase 3: empty rag (rag=[], rule=[A,B,C])")
    result = fuse([], [A, B, C], k_final=3)
    print(f"  result:   {[s['title'] for s in result]}")
    print(f"  expected: ['A', 'B', 'C']")

    # Case 4: both empty → no crash, empty result
    print("\nCase 4: both empty")
    result = fuse([], [], k_final=10)
    print(f"  result:   {result}")
    print(f"  expected: []")

    # Case 5: k_final trims correctly
    print("\nCase 5: k_final=2 (rag=[A,B,C,D,E], rule=[])")
    result = fuse([A, B, C, D, E], [], k_final=2)
    print(f"  result:   {[s['title'] for s in result]}")
    print(f"  expected: ['A', 'B']")



# can drop if numeric dimensions too far