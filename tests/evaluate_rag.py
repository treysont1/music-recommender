"""
Evaluation script for the RAG retrieval pipeline.

Runs predefined queries against the index, prints similarity scores,
pass/fail status per query, and a final summary.

Usage:
    python tests/evaluate_rag.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag import _retrieve, _validate_query, LOW_CONFIDENCE_THRESHOLD

EVAL_QUERIES = [
    # (query, description)
    ("sad slow songs for a late night drive",   "mood-based"),
    ("high energy workout music",               "activity-based"),
    ("chill indie music for a Sunday morning",  "vibe-based"),
    ("upbeat pop songs to dance to",            "genre-based"),
    ("jazz for a rainy afternoon",              "genre + mood"),
    ("what is the capital of france",           "off-topic — should be rejected"),
    ("",                                        "empty — should be rejected"),
]


def run_evaluation() -> None:
    print("=" * 68)
    print("  RAG Evaluation Report")
    print("=" * 68)

    passed = 0
    total = len(EVAL_QUERIES)

    for query, description in EVAL_QUERIES:
        validation_error = _validate_query(query)

        if validation_error:
            # Off-topic / empty queries should be caught by validation
            status = "PASS (rejected by guardrail)"
            passed += 1
            print(f"\n  Query      : {repr(query)}")
            print(f"  Type       : {description}")
            print(f"  Status     : {status}")
            print(f"  Reason     : {validation_error}")
            continue

        songs, similarities = _retrieve(query)
        avg_sim = round(sum(similarities) / len(similarities), 3)
        top_sim = max(similarities)
        low_sim = min(similarities)
        above_threshold = avg_sim >= LOW_CONFIDENCE_THRESHOLD

        status = "PASS" if above_threshold else "FAIL (low confidence)"
        if above_threshold:
            passed += 1

        print(f"\n  Query      : {repr(query)}")
        print(f"  Type       : {description}")
        print(f"  Avg sim    : {avg_sim}  |  Range: {low_sim}–{top_sim}")
        print(f"  Top result : \"{songs[0]['title']}\" by {songs[0]['artist']}")
        print(f"  Status     : {status}")

    print("\n" + "=" * 68)
    print(f"  Result: {passed}/{total} passed  |  Threshold: {LOW_CONFIDENCE_THRESHOLD}")
    print("=" * 68)


if __name__ == "__main__":
    run_evaluation()
