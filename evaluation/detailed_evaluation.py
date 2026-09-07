import json
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher
import re

RESULTS_DIR = Path(__file__).parent / "results"


def similarity(expected, actual):
    return SequenceMatcher(
        None,
        expected.lower().strip(),
        actual.lower().strip()
    ).ratio()


def is_safe_outside_kb(result):
    """Check whether the model correctly refused unsupported information."""

    answer = result.get("answer", "").lower()

    safe_phrases = [
        "not specified",
        "not explicitly stated",
        "not available",
        "not provided",
        "does not provide",
        "does not specify",
        "not mentioned",
        "not stated",
        "information is not available",
    ]

    return any(phrase in answer for phrase in safe_phrases)


def classify_correctness(result):
    """
    Preliminary classification.

    Similarity is used only as a starting point.
    Question-level results should be manually reviewed.
    """

    category = result.get("category", "")

    if category == "Outside Knowledge Base":
        if is_safe_outside_kb(result):
            return "SAFE_REFUSAL"

    score = similarity(
        result.get("expected_answer", ""),
        result.get("answer", "")
    )

    if score >= 0.80:
        return "CORRECT"

    elif score >= 0.50:
        return "PARTIAL"

    else:
        return "WRONG"


def potential_hallucination(result):
    """
    Conservative hallucination detector.

    It checks whether the answer introduces numerical
    information when the expected answer says that the
    exact information is not available.

    This is only a FLAG for manual review.
    It does NOT prove hallucination.
    """

    expected = result.get("expected_answer", "").lower()
    answer = result.get("answer", "").lower()

    unspecified = [
        "not specified",
        "not explicitly stated",
        "not available",
        "does not specify",
        "not provided"
    ]

    if not any(x in expected for x in unspecified):
        return False

    # Percentage claims
    if re.search(r"\b\d+(\.\d+)?\s*%", answer):
        return True

    # Time/quantity claims
    if re.search(
        r"\b\d+\s*(days?|weeks?|months?|hours?)\b",
        answer
    ):
        return True

    return False


def retrieval_quality(result):
    """
    Simple retrieval analysis based on the retrieved chunks.

    The top retrieved chunk is treated as relevant when it
    contains meaningful overlap with the expected answer.
    """

    expected = result.get("expected_answer", "").lower()

    contexts = result.get("retrieved_context", [])

    if not contexts:
        return "NO_CONTEXT"

    top_chunk = contexts[0].get("chunk", "").lower()

    expected_words = {
        word
        for word in re.findall(r"\b[a-zA-Z]{4,}\b", expected)
    }

    context_words = {
        word
        for word in re.findall(r"\b[a-zA-Z]{4,}\b", top_chunk)
    }

    overlap = expected_words.intersection(context_words)

    if len(overlap) >= 3:
        return "RELEVANT"

    return "POSSIBLY_IRRELEVANT"


for file in sorted(RESULTS_DIR.glob("*.json")):

    with open(file, "r", encoding="utf-8") as f:
        results = json.load(f)

    total = len(results)

    classifications = [
        classify_correctness(r)
        for r in results
    ]

    hallucinations = [
        potential_hallucination(r)
        for r in results
    ]

    retrievals = [
        retrieval_quality(r)
        for r in results
    ]

    scores = [
        similarity(
            r.get("expected_answer", ""),
            r.get("answer", "")
        )
        for r in results
    ]

    latencies = [
        r.get("latency_seconds", 0)
        for r in results
        if "latency_seconds" in r
    ]

    counts = Counter(classifications)
    retrieval_counts = Counter(retrievals)

    print("=" * 70)
    print(f"MODEL: {file.stem}")
    print("=" * 70)

    print(f"Questions evaluated : {total}")

    print("\nQUALITY")

    print(
        f"Correct             : "
        f"{counts['CORRECT']}/{total} "
        f"({counts['CORRECT']/total*100:.1f}%)"
    )

    print(
        f"Partial             : "
        f"{counts['PARTIAL']}/{total} "
        f"({counts['PARTIAL']/total*100:.1f}%)"
    )

    print(
        f"Wrong               : "
        f"{counts['WRONG']}/{total} "
        f"({counts['WRONG']/total*100:.1f}%)"
    )

    print(
        f"Safe refusal        : "
        f"{counts['SAFE_REFUSAL']}/{total} "
        f"({counts['SAFE_REFUSAL']/total*100:.1f}%)"
    )

    hallucination_count = sum(hallucinations)

    print(
        f"Potential halluc.   : "
        f"{hallucination_count}/{total} "
        f"({hallucination_count/total*100:.1f}%)"
    )

    print("\nRETRIEVAL")

    print(
        f"Relevant top context: "
        f"{retrieval_counts['RELEVANT']}/{total}"
    )

    print(
        f"Possibly irrelevant : "
        f"{retrieval_counts['POSSIBLY_IRRELEVANT']}/{total}"
    )

    print(
        f"No context          : "
        f"{retrieval_counts['NO_CONTEXT']}/{total}"
    )

    print("\nPERFORMANCE")

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

    print("\nLEXICAL SIMILARITY")

    print(
        f"Average similarity  : "
        f"{sum(scores)/len(scores):.3f}"
    )

    print(
        f"Minimum similarity  : "
        f"{min(scores):.3f}"
    )

    print(
        f"Maximum similarity  : "
        f"{max(scores):.3f}"
    )

    print("\nQUESTION-LEVEL REVIEW")
    print("-" * 70)

    for result, classification, hallucination, retrieval, score in zip(
        results,
        classifications,
        hallucinations,
        retrievals,
        scores
    ):

        print(
            f"Q{result['id']:02d} | "
            f"{classification:<13} | "
            f"retrieval={retrieval:<18} | "
            f"similarity={score:.3f} | "
            f"potential_hallucination="
            f"{'YES' if hallucination else 'NO'}"
        )

    print()

print("=" * 70)
print("IMPORTANT:")
print("Potential hallucination is a conservative automatic FLAG.")
print("It must be manually verified against the retrieved context.")
print("=" * 70)