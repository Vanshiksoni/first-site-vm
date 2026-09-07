import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

selected_questions = ["7", "21", "28", "29", "30"]

for file in RESULTS_DIR.glob("*.json"):

    with open(file, "r", encoding="utf-8") as f:
        results = json.load(f)

    print("\n" + "=" * 80)
    print("MODEL:", file.stem)
    print("=" * 80)

    for r in results:

        if str(r["id"]) not in selected_questions:
            continue

        print("\n" + "-" * 80)
        print("QUESTION:", r["question"])
        print("\nRETRIEVED CONTEXT:")

        for context in r.get("retrieved_context", []):
            print(context)

        print("\nLLM ANSWER:")
        print(r["answer"])

        print("\n" + "-" * 80)