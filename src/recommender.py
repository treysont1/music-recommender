import csv
from typing import List, Dict, Tuple
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEMPO_MIN = 60
TEMPO_MAX = 178

WEIGHTS = {
    "genre":        3.0,
    "mood":         2.0,
    "energy":       2.0,
    "acousticness": 1.0,
    "valence":      1.0,
    "tempo_bpm":    1.0,
}

# Partial credit for related genres. Keys are sorted tuples so order does
# not matter. Any pair not listed receives 0.0 (unrelated).
GENRE_SIMILARITY = {
    ("edm",       "hip-hop"):   0.3,
    ("hip-hop",   "pop"):       0.4,
    ("hip-hop",   "r&b"):       0.7,
    ("dream pop", "indie pop"): 0.7,
    ("indie pop", "pop"):       0.6,
    ("indie pop", "rock"):      0.3,
    ("dream pop", "pop"):       0.4,
    ("ambient",   "dream pop"): 0.5,
    ("ambient",   "lofi"):      0.6,
    ("jazz",      "lofi"):      0.5,
    ("dream pop", "lofi"):      0.3,
    ("metal",     "rock"):      0.7,
    ("edm",       "synthwave"): 0.5,
    ("pop",       "synthwave"): 0.3,
    ("blues",     "jazz"):      0.6,
    ("classical", "jazz"):      0.3,
    ("country",   "folk"):      0.6,
    ("blues",     "folk"):      0.4,
    ("blues",     "country"):   0.5,
    ("ambient",   "classical"): 0.4,
    ("edm",       "pop"):       0.3,
    ("pop",       "r&b"):       0.4,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


# ---------------------------------------------------------------------------
# Internal scoring helpers
# ---------------------------------------------------------------------------

def _genre_score(song_genre: str, user_genre: str) -> float:
    """Returns 1.0 for an exact match, a partial value for related genres,
    or 0.0 for unrelated genres."""
    if song_genre == user_genre:
        return 1.0
    key = tuple(sorted([song_genre, user_genre]))
    return GENRE_SIMILARITY.get(key, 0.0)


def _normalize_tempo(tempo: float) -> float:
    """Scales a raw BPM value to the 0.0–1.0 range using catalog bounds."""
    return (tempo - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN)


def _build_explanation(song_genre, song_mood, song_energy, song_acousticness,
                       song_tempo_bpm, user_prefs: Dict,
                       genre_s, mood_s, energy_s, acoustic_s) -> str:
    """Builds a human-readable explanation of the top reasons for a score."""
    reasons = []

    if "genre" in user_prefs:
        if genre_s == 1.0:
            reasons.append(f"exact genre match ({song_genre})")
        elif genre_s >= 0.5:
            reasons.append(
                f"closely related genre ({song_genre} ~ {user_prefs['genre']}, "
                f"similarity {genre_s})"
            )
        elif genre_s > 0:
            reasons.append(
                f"loosely related genre ({song_genre} ~ {user_prefs['genre']}, "
                f"similarity {genre_s})"
            )

    if "mood" in user_prefs and mood_s == 1.0:
        reasons.append(f"mood matches ({song_mood})")

    if "energy" in user_prefs and energy_s >= 0.85:
        reasons.append(
            f"energy is close ({song_energy} vs target {user_prefs['energy']})"
        )

    if "acousticness" in user_prefs and acoustic_s >= 0.85:
        reasons.append(
            f"acousticness fits preference ({song_acousticness})"
        )

    return ", ".join(reasons) if reasons else "numeric features are a reasonable overall fit"


# ---------------------------------------------------------------------------
# Core scoring function (shared by both functional and OOP paths)
# ---------------------------------------------------------------------------

def _compute_score(song_genre: str, song_mood: str, song_energy: float,
                   song_acousticness: float, song_valence: float,
                   song_tempo_bpm: float, user_prefs: Dict) -> Tuple[float, str]:
    """
    Computes a weighted score in the range 0.0–1.0 and an explanation string.
    """
    genre_s    = _genre_score(song_genre, user_prefs.get("genre", ""))
    mood_s     = 1.0 if song_mood == user_prefs.get("mood") else 0.0
    energy_s   = 1.0 - abs(song_energy - user_prefs.get("energy", song_energy))
    acoustic_s = 1.0 - abs(song_acousticness - user_prefs.get("acousticness", song_acousticness))
    valence_s  = 1.0 - abs(song_valence - user_prefs.get("valence", song_valence))

    song_tempo_norm = _normalize_tempo(song_tempo_bpm)
    user_tempo_norm = _normalize_tempo(user_prefs.get("tempo_bpm", song_tempo_bpm))
    tempo_s = 1.0 - abs(song_tempo_norm - user_tempo_norm)


    raw = (
        genre_s    * WEIGHTS["genre"]        +
        mood_s     * WEIGHTS["mood"]         +
        energy_s   * WEIGHTS["energy"]       +
        acoustic_s * WEIGHTS["acousticness"] +
        valence_s  * WEIGHTS["valence"]      +
        tempo_s    * WEIGHTS["tempo_bpm"]
    )

    score = round(raw / 10.0, 3)

    explanation = _build_explanation(
        song_genre, song_mood, song_energy, song_acousticness,
        song_tempo_bpm, user_prefs, genre_s, mood_s, energy_s, acoustic_s
    )

    return score, explanation


# ---------------------------------------------------------------------------
# OOP interface
# ---------------------------------------------------------------------------

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def _profile_to_prefs(self, user: UserProfile) -> Dict:
        """Converts a UserProfile into the dict format _compute_score expects."""
        return {
            "genre":        user.favorite_genre,
            "mood":         user.favorite_mood,
            "energy":       user.target_energy,
            "acousticness": 0.2 if not user.likes_acoustic else 0.8,
            "valence":      0.7,
            "tempo_bpm":    100,
        }

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        prefs = self._profile_to_prefs(user)
        scored = [
            (song, _compute_score(
                song.genre, song.mood, song.energy,
                song.acousticness, song.valence, song.tempo_bpm, prefs
            )[0])
            for song in self.songs
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [song for song, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        prefs = self._profile_to_prefs(user)
        _, explanation = _compute_score(
            song.genre, song.mood, song.energy,
            song.acousticness, song.valence, song.tempo_bpm, prefs
        )
        return explanation


# ---------------------------------------------------------------------------
# Functional interface (used by pipeline.py via _retrieve_rules)
# ---------------------------------------------------------------------------

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by pipeline.py
    """
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "title":        row["title"],
                "artist":       row["artist"],
                "genre":        row["genre"],
                "mood":         row["mood"],
                "energy":       float(row["energy"]),
                "tempo_bpm":    float(row["tempo_bpm"]),
                "valence":      float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
            })
    return songs


def recommend_songs(user_prefs: Dict, songs: List[Dict],
                    k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by pipeline.py
    Returns a list of (song_dict, score, explanation) tuples, best first.
    """
    scored = []
    for song in songs:
        score, explanation = _compute_score(
            song["genre"], song["mood"], song["energy"],
            song["acousticness"], song["valence"], song["tempo_bpm"],
            user_prefs
        )
        scored.append((song, score, explanation))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
