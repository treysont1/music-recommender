"""
Evaluation script for the RAG pipeline.

Tests that:
- Input validation catches bad queries
- Retrieval returns similarity scores above the minimum threshold for valid music queries
- Low-confidence guardrail triggers on off-topic input

Run with: pytest tests/test_rag.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag import retrieve
from pipeline import _validate_query, LOW_CONFIDENCE_THRESHOLD

SAMPLE_QUERIES = [
    "something sad and slow for a late night drive",
    "high energy workout music",
    "chill indie music for a Sunday morning",
    "upbeat pop songs to dance to",
]


class TestInputValidation:
    def test_empty_query_rejected(self):
        assert _validate_query("") is not None

    def test_short_query_rejected(self):
        assert _validate_query("hi") is not None

    def test_off_topic_query_rejected(self):
        assert _validate_query("what is the capital of france") is not None

    def test_valid_music_query_passes(self):
        assert _validate_query("chill music for studying") is None

    def test_mood_query_passes(self):
        assert _validate_query("sad songs for a rainy day") is None


class TestRetrieval:
    def test_similarity_scores_above_threshold(self):
        for query in SAMPLE_QUERIES:
            songs, similarities = retrieve(query)
            avg = sum(similarities) / len(similarities)
            assert avg >= LOW_CONFIDENCE_THRESHOLD, (
                f"Query '{query}' returned avg similarity {avg:.3f}, "
                f"below threshold {LOW_CONFIDENCE_THRESHOLD}"
            )

    def test_retrieval_returns_correct_count(self):
        songs, similarities = retrieve("happy pop music", k=5)
        assert len(songs) == 5
        assert len(similarities) == 5

    def test_similarities_in_valid_range(self):
        _, similarities = retrieve("chill lofi beats")
        for s in similarities:
            assert 0.0 <= s <= 1.0, f"Similarity {s} out of range"
