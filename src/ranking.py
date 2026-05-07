import re

# song_genre: str, 
# song_mood: str, 
# song_energy: float,
# song_acousticness: float, 
# song_valence: float,
# song_tempo_bpm:


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




def extract_preferences(query: str) -> dict:
    preferences = {}



def fuse(rag_results, rule_results, K: int) -> list[dict]:
    pass






# can drop if numeric dimensions too far