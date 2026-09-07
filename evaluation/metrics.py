import json
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher

RESULTS_DIR = Path(__file__).parent / "results"


def similarity(expected, actual):
    return SequenceMatcher(
        None,
        expected.lower().strip(),
        actual.lower().strip()
    ).ratio()


def classify_answer(result):
    """
    Classifies the model response using the expected answer,
    question category, and response text.

    This is a rule-based evaluation, so flagged cases should
    be manually reviewed before being reported as hallucinations.
    """

    expected = result.get("expected_answer", "").lower().strip()
    actual = result.get("answer", "").lower().strip()
    category = result.get("category", "")

    score = similarity(expected, actual)

    # Outside-KB questions
    if category == "Outside Knowledge Base":
        safe_phrases = [
            "not specified",
            "not explicitly stated",
            "not available",
            "not provided",
            "does not provide",
            "does not specify",
            "information is not available",
            "not mentioned",
            "not stated"
        ]

        if any(phrase in actual for phrase in safe_phrases):
            return "SAFE_REFUSAL"

    # Strong textual match
    if score >= 0.80:
        return "CORRECT"

    # Moderate similarity
    if score >= 0.50:
        return "PARTIAL"

    # Very low similarity
    return "WRONG"


def detect_hallucination(result):
    """
    Conservative hallucination proxy.

    Flags answers that contain unsupported numerical claims
    or obvious fabricated specifics.

    This is NOT a definitive hallucination detector.
    """

    actual = result.get("answer", "").lower()

    # Numerical claims are suspicious when the expected answer
    # explicitly says the information is not specified.
    expected = result.get("expected_answer", "").lower()

    unspecified = [
        "not specified",
        "not explicitly stated",
        "not available",
        "does not specify"
    ]

    if any(x in expected for x in unspecified):

        # Look for percentages or quantities
        import re

        if re.search(r"\b\d+(\.\d+)?\s*%", actual):
            return True

        if re.search(
            r"\b\d+\s*(days?|weeks?|months?|hours?)\b",
            actual
        ):
            return True

    return False


for file in sorted(RESULTS_DIR.glob("*.json")):

    with open(file, "r", encoding="utf-8") as f:
        results = json.load(f)

    total = len(results)

    rag_used = sum(
        r.get("rag_used", False)
        for r in results
    )

    llm_used = sum(
        r.get("llm_used", False)
        for r in results
    )

    latencies = [
        r["latency_seconds"]
        for r in results
        if "latency_seconds" in r
    ]

    scores = [
        similarity(
            r.get("expected_answer", ""),
            r.get("answer", "")
        )
        for r in results
    ]

    classifications = [
        classify_answer(r)
        for r in results
    ]

    hallucinations = [
        detect_hallucination(r)
        for r in results
    ]

    print("=" * 60)
    print(f"Model: {file.stem}")
    print("=" * 60)

    print(f"Questions evaluated : {total}")
    print(f"RAG used            : {rag_used}/{total}")
    print(f"LLM used            : {llm_used}/{total}")

    if latencies:
        print(
            f"Average latency     : "
            f"{sum(latencies)/len(latencies):.3f} seconds"
        )
        print(
            f"Minimum latency     : "
            f"{min(latencies):.3f} seconds"
        )
        print(
            f"Maximum latency     : "
            f"{max(latencies):.3f} seconds"
        )

    if scores:
        print(
            f"Average similarity  : "
            f"{sum(scores)/len(scores):.3f}"
        )

    counts = Counter(classifications)

    correct = counts["CORRECT"]
    partial = counts["PARTIAL"]
    wrong = counts["WRONG"]
    safe = counts["SAFE_REFUSAL"]

    print(
        f"Correct / Exact     : "
        f"{correct}/{total} ({correct/total*100:.1f}%)"
    )

    print(
        f"Partial             : "
        f"{partial}/{total} ({partial/total*100:.1f}%)"
    )

    print(
        f"Wrong               : "
        f"{wrong}/{total} ({wrong/total*100:.1f}%)"
    )

    print(
        f"Safe refusal        : "
        f"{safe}/{total} ({safe/total*100:.1f}%)"
    )

    hallucination_count = sum(hallucinations)

    print(
        f"Hallucination proxy : "
        f"{hallucination_count}/{total} "
        f"({hallucination_count/total*100:.1f}%)"
    )

    outside_kb = [
        r for r in results
        if r.get("category") == "Outside Knowledge Base"
    ]

    if outside_kb:
        safe_outside = sum(
            classify_answer(r) == "SAFE_REFUSAL"
            for r in outside_kb
        )

        print(
            f"Outside-KB safe     : "
            f"{safe_outside}/{len(outside_kb)}"
        )

    print("\nCategories:")

    categories = Counter(
        r["category"]
        for r in results
    )

    for category, count in categories.items():

        category_results = [
            r for r in results
            if r["category"] == category
        ]

        category_scores = [
            similarity(
                r.get("expected_answer", ""),
                r.get("answer", "")
            )
            for r in category_results
        ]

        print(
            f"  {category}: {count} "
            f"(avg similarity: "
            f"{sum(category_scores)/len(category_scores):.3f})"
        )

    print("\nQuestion-level evaluation:")

    for result, classification, hallucination in zip(
        results,
        classifications,
        hallucinations
    ):

        score = similarity(
            result.get("expected_answer", ""),
            result.get("answer", "")
        )

        print(
            f"  Q{result['id']:02d} | "
            f"{classification:<12} | "
            f"similarity={score:.3f} | "
            f"hallucination="
            f"{'YES' if hallucination else 'NO'}"
        )

    print()