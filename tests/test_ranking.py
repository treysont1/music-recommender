"""
Tests for ranking.extract_preferences().

Covers:
- Sanity cases: single keywords, categorical agreement/conflict, empty input.
- Edge cases: punctuation, case sensitivity, whitespace, word boundaries, rounding.

Run with: pytest tests/test_ranking.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ranking import extract_preferences, fuse, RRF_K


class TestSanityCases:
    def test_single_keyword(self):
        result = extract_preferences("chill")
        assert result == {"mood": "chill", "valence": 0.35, "energy": 0.3}

    def test_categorical_agreement(self):
        # "happy" and "upbeat" both set mood: "happy" — should NOT be dropped
        result = extract_preferences("happy upbeat")
        assert result["mood"] == "happy"

    def test_categorical_conflict_drops_dimension(self):
        # "chill" → mood: "chill", "intense" → mood: "intense" → conflict, drop
        result = extract_preferences("chill intense")
        assert "mood" not in result
        assert result["energy"] == 0.575    # avg(0.3, 0.85)
        assert result["valence"] == 0.35    # only chill sets valence

    def test_no_match_returns_empty(self):
        result = extract_preferences("hello there")
        assert result == {}

    def test_empty_string_returns_empty(self):
        result = extract_preferences("")
        assert result == {}


class TestEdgeCases:
    def test_punctuation_ignored(self):
        result = extract_preferences("chill, workout!")
        expected = extract_preferences("chill workout")
        assert result == expected

    def test_case_insensitive(self):
        result = extract_preferences("CHILL Workout")
        expected = extract_preferences("chill workout")
        assert result == expected

    def test_extra_whitespace(self):
        result = extract_preferences("  chill\n\nworkout  ")
        expected = extract_preferences("chill workout")
        assert result == expected

    def test_word_boundaries_prevent_substring_match(self):
        # "rapid" contains "rap" but should NOT trigger the rap → hip-hop keyword
        result = extract_preferences("rapid acoustic")
        assert "genre" not in result
        assert result.get("acousticness") == 0.85

    def test_numeric_rounding(self):
        # chill, calm, mellow all have energy values: 0.3, 0.2, 0.3
        # avg = 0.8 / 3 = 0.2666...  → rounds to 0.267
        result = extract_preferences("chill calm mellow")
        assert result["energy"] == 0.267


def _song(title, artist="X"):
    """Helper: minimal song dict for fuse() tests."""
    return {"title": title, "artist": artist}


class TestFuseSanity:
    def test_overlap_outranks_single_channel(self):
        # Songs in both lists should beat songs in only one
        a, b, c, d = _song("A"), _song("B"), _song("C"), _song("D")
        result = fuse([a, b, c], [b, c, d], k_final=4)
        titles = [s["title"] for s in result]
        assert titles == ["B", "C", "A", "D"]

    def test_empty_rule_falls_back_to_rag(self):
        # Sparse-prefs degradation: no rule signal → RAG ordering preserved
        a, b, c = _song("A"), _song("B"), _song("C")
        result = fuse([a, b, c], [], k_final=3)
        titles = [s["title"] for s in result]
        assert titles == ["A", "B", "C"]

    def test_empty_rag_falls_back_to_rule(self):
        a, b, c = _song("A"), _song("B"), _song("C")
        result = fuse([], [a, b, c], k_final=3)
        titles = [s["title"] for s in result]
        assert titles == ["A", "B", "C"]

    def test_both_empty_returns_empty(self):
        assert fuse([], [], k_final=10) == []

    def test_k_final_trims_results(self):
        a, b, c, d, e = (_song(t) for t in "ABCDE")
        result = fuse([a, b, c, d, e], [], k_final=2)
        assert len(result) == 2
        assert [s["title"] for s in result] == ["A", "B"]


class TestFuseEdgeCases:
    def test_identical_lists_preserve_order(self):
        # Both channels agree perfectly → output matches input order
        songs = [_song(t) for t in "ABCDE"]
        result = fuse(songs, songs, k_final=5)
        assert [s["title"] for s in result] == ["A", "B", "C", "D", "E"]

    def test_returns_song_dicts_not_ids(self):
        # Output should be list[dict], not list[str]
        a, b = _song("A"), _song("B")
        result = fuse([a, b], [], k_final=2)
        assert all(isinstance(s, dict) for s in result)
        assert result[0] == a

    def test_k_final_larger_than_unique_songs(self):
        # If k_final exceeds available songs, return all without padding
        a, b = _song("A"), _song("B")
        result = fuse([a, b], [], k_final=10)
        assert len(result) == 2

    def test_disjoint_lists(self):
        # No overlap between channels — both contribute, ordering by rank
        a, b = _song("A"), _song("B")
        c, d = _song("C"), _song("D")
        result = fuse([a, b], [c, d], k_final=4)
        # All four songs scored from one channel only.
        # Within RAG: A=1/61, B=1/62. Within rule: C=1/61, D=1/62.
        # A and C tie at 1/61, B and D tie at 1/62 — sort is stable but tie order undefined.
        # Just check membership and that rank-1s outrank rank-2s.
        titles = [s["title"] for s in result]
        assert set(titles[:2]) == {"A", "C"}
        assert set(titles[2:]) == {"B", "D"}

    def test_single_song_each_list(self):
        # Minimal valid input
        a = _song("A")
        result = fuse([a], [a], k_final=10)
        assert result == [a]
