import json
import time
import sys
import requests
from pathlib import Path


APPLICATION_URL = "http://localhost:8002/ask"

DATASET_PATH = Path(__file__).parent / "evaluation_dataset.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_results(output_path, results):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )


def evaluate_model(model_name, start_question=1, end_question=None):

    dataset = load_dataset()

    RESULTS_DIR.mkdir(exist_ok=True)

    safe_model_name = (
        model_name
        .replace(":", "_")
        .replace("/", "_")
    )

    output_path = RESULTS_DIR / f"{safe_model_name}.json"

    # Load previous results if they already exist
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as file:
                results = json.load(file)

            print(f"Existing results found: {len(results)}")

        except Exception:
            results = []

    else:
        results = []

    completed_ids = {
        result["id"]
        for result in results
    }

    total_questions = len(dataset)

    if end_question is None:
        end_question = total_questions

    print("=" * 70)
    print(f"Evaluating model: {model_name}")
    print(f"Questions {start_question} to {end_question}")
    print(f"Total dataset questions: {total_questions}")
    print("=" * 70)

    for index in range(start_question, end_question + 1):

        item = dataset[index - 1]

        question_id = item["id"]
        question = item["question"]

        # Skip questions that were already saved
        if question_id in completed_ids:
            print(f"\n[{index}/{total_questions}] Already completed - skipping")
            continue

        print(f"\n[{index}/{total_questions}] {question}")

        start_time = time.perf_counter()

        try:

            response = requests.post(
                APPLICATION_URL,
                json={
                    "question": question
                },
                timeout=180
            )

            response.raise_for_status()

            data = response.json()

            end_time = time.perf_counter()

            latency = end_time - start_time

            result = {
                "id": question_id,
                "category": item["category"],
                "question": question,
                "expected_answer": item["expected_answer"],
                "model": data.get("model", model_name),
                "answer": data.get("answer", ""),
                "rag_used": data.get("rag_used", False),
                "llm_used": data.get("llm_used", False),
                "retrieved_context": data.get(
                    "retrieved_context",
                    []
                ),
                "latency_seconds": round(latency, 3)
            }

            results.append(result)

            print(f"Model: {result['model']}")
            print(f"Latency: {result['latency_seconds']} seconds")
            print(f"RAG used: {result['rag_used']}")
            print(f"LLM used: {result['llm_used']}")

        except Exception as error:

            end_time = time.perf_counter()

            latency = end_time - start_time

            print(f"ERROR: {error}")

            result = {
                "id": question_id,
                "category": item["category"],
                "question": question,
                "expected_answer": item["expected_answer"],
                "model": model_name,
                "answer": "",
                "rag_used": False,
                "llm_used": False,
                "retrieved_context": [],
                "latency_seconds": round(latency, 3),
                "error": str(error)
            }

            results.append(result)

        # IMPORTANT:
        # Save immediately after every question
        save_results(output_path, results)

        print(f"Saved checkpoint: {output_path}")

    print("\n" + "=" * 70)
    print("EVALUATION BATCH COMPLETE")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Results saved to: {output_path}")
    print(f"Total saved results: {len(results)}")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage: python3 evaluate_model.py MODEL_NAME [START] [END]"
        )
        print(
            "Example: python3 evaluate_model.py codellama 1 5"
        )
        sys.exit(1)

    model = sys.argv[1]

    start = int(sys.argv[2]) if len(sys.argv) >= 3 else 1

    end = int(sys.argv[3]) if len(sys.argv) >= 4 else None

    evaluate_model(
        model,
        start_question=start,
        end_question=end
    )
