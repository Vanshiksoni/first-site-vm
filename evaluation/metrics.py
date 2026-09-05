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


result_files = list(RESULTS_DIR.glob("*.json"))

for file in result_files:

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
        print(
            f"Minimum similarity  : "
            f"{min(scores):.3f}"
        )
        print(
            f"Maximum similarity  : "
            f"{max(scores):.3f}"
        )

    print("\nCategories:")

    categories = Counter(
        r["category"]
        for r in results
    )

    for category, count in categories.items():
        print(f"  {category}: {count}")

    print()