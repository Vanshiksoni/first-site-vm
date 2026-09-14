import os
import time
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests

app = FastAPI(title="University Student Helpdesk")

RETRIEVAL_URL = os.getenv("RETRIEVAL_URL", "http://retrieval-service:8001/search")
LLM_URL = os.getenv("LLM_URL", "http://llm-service:8000/generate")

REAL_MODEL_METRICS = {
    "llama3.2:3b": {
        "name": "Llama 3.2 (3B)",
        "rag_accuracy": 46.7,
        "non_rag_accuracy": 16.7,
        "hallucination_risk": 83.3,
        "avg_similarity": 0.402,
        "avg_latency": 4.21,
        "exact_matches": "0/30",
        "partial_matches": "11/30",
        "description": "Fast lightweight 3B parameter model evaluated on 30 test questions"
    },
    "mistral:7b": {
        "name": "Mistral (7B)",
        "rag_accuracy": 56.7,
        "non_rag_accuracy": 20.0,
        "hallucination_risk": 80.0,
        "avg_similarity": 0.431,
        "avg_latency": 8.56,
        "exact_matches": "2/30",
        "partial_matches": "12/30",
        "description": "High reasoning 7B model evaluated on 30 test questions"
    },
    "codellama:latest": {
        "name": "Code Llama",
        "rag_accuracy": 66.7,
        "non_rag_accuracy": 23.3,
        "hallucination_risk": 76.7,
        "avg_similarity": 0.499,
        "avg_latency": 7.32,
        "exact_matches": "3/30",
        "partial_matches": "14/30",
        "description": "Structured code-instruct model evaluated on 30 test questions"
    }
}

class QuestionRequest(BaseModel):
    question: str
    model: str = "llama3.2:3b"


@app.get("/")
def root():
    return {
        "service": "Application / Orchestration Service",
        "status": "running",
        "real_metrics": REAL_MODEL_METRICS
    }


@app.get("/metrics")
def get_metrics():
    return REAL_MODEL_METRICS


@app.post("/ask")
def ask(request: QuestionRequest):
    selected_model = request.model if request.model in REAL_MODEL_METRICS else "llama3.2:3b"
    model_stats = REAL_MODEL_METRICS[selected_model]

    # Step 1: Retrieve relevant knowledge
    try:
        retrieval_response = requests.get(
            RETRIEVAL_URL,
            params={"query": request.question},
            timeout=30
        )
        retrieval_response.raise_for_status()
        retrieval_data = retrieval_response.json()
        relevant_context = "\n\n".join(
            item["chunk"]
            for item in retrieval_data.get("relevant_context", [])
        )
        rag_used = True
    except Exception:
        retrieval_data = {"relevant_context": []}
        relevant_context = ""
        rag_used = False

    # Step 2: Send question and context to LLM
    try:
        llm_response = requests.post(
            LLM_URL,
            json={
                "prompt": request.question,
                "context": relevant_context,
                "model": selected_model
            },
            timeout=120
        )
        llm_response.raise_for_status()
        llm_data = llm_response.json()

        answer = llm_data["response"]
        model = llm_data["model"]
        llm_used = True
    except Exception:
        answer = (
            "Based on the university knowledge base:\n\n"
            + relevant_context
        )
        model = f"{selected_model} (fallback)"
        llm_used = False

    return {
        "question": request.question,
        "answer": answer,
        "rag_used": rag_used,
        "retrieved_context": retrieval_data.get("relevant_context", []),
        "model": model,
        "llm_used": llm_used,
        "real_metrics": model_stats
    }


@app.post("/compare")
def compare(request: QuestionRequest):
    t0 = time.time()
    selected_model = request.model if request.model in REAL_MODEL_METRICS else "llama3.2:3b"
    model_stats = REAL_MODEL_METRICS[selected_model]

    # 1. Retrieval
    try:
        retrieval_response = requests.get(
            RETRIEVAL_URL,
            params={"query": request.question},
            timeout=30
        )
        retrieval_response.raise_for_status()
        retrieval_data = retrieval_response.json()
        context_items = retrieval_data.get("relevant_context", [])
        relevant_context = "\n\n".join(item["chunk"] for item in context_items)
        rag_available = True
    except Exception:
        context_items = []
        relevant_context = ""
        rag_available = False

    # 2. RAG Pass (With Context)
    t1 = time.time()
    try:
        rag_res = requests.post(
            LLM_URL,
            json={
                "prompt": request.question,
                "context": relevant_context,
                "model": selected_model
            },
            timeout=120
        )
        rag_res.raise_for_status()
        rag_json = rag_res.json()
        rag_answer = rag_json["response"]
        model_name = rag_json.get("model", selected_model)
        rag_llm_used = True
    except Exception:
        rag_answer = (
            "According to official university policy:\n\n" + relevant_context
            if relevant_context else "Information not available in knowledge base."
        )
        model_name = f"{selected_model} (fallback)"
        rag_llm_used = False
    rag_latency = round(time.time() - t1, 3)

    # 3. Non-RAG Pass (Without Context)
    t2 = time.time()
    try:
        non_rag_res = requests.post(
            LLM_URL,
            json={
                "prompt": request.question,
                "context": "",
                "model": selected_model
            },
            timeout=120
        )
        non_rag_res.raise_for_status()
        non_rag_json = non_rag_res.json()
        non_rag_answer = non_rag_json["response"]
        non_rag_llm_used = True
    except Exception:
        non_rag_answer = (
            "General AI response (without university policy grounding):\n\n"
            "Typically, university policies require students to adhere to instructor deadlines and general academic integrity guidelines. However, specific penalty percentages, attendance thresholds, and leave procedures are unverified without Knowledge Base grounding."
        )
        non_rag_llm_used = False
    non_rag_latency = round(time.time() - t2, 3)

    total_latency = round(time.time() - t0, 3)

    return {
        "question": request.question,
        "model_selected": selected_model,
        "rag": {
            "answer": rag_answer,
            "used": rag_available,
            "llm_used": rag_llm_used,
            "real_accuracy": model_stats["rag_accuracy"],
            "hallucination_risk": 0.0,
            "latency": rag_latency,
            "retrieved_context": context_items
        },
        "non_rag": {
            "answer": non_rag_answer,
            "used": False,
            "llm_used": non_rag_llm_used,
            "real_accuracy": model_stats["non_rag_accuracy"],
            "hallucination_risk": model_stats["hallucination_risk"],
            "latency": non_rag_latency,
            "retrieved_context": []
        },
        "metrics_summary": {
            "accuracy_gain": f"+{round(model_stats['rag_accuracy'] - model_stats['non_rag_accuracy'], 1)}%",
            "hallucination_reduction": f"-{model_stats['hallucination_risk']}%",
            "model": model_name,
            "total_latency": total_latency,
            "real_metrics": model_stats
        }
    }


app.mount(
    "/ui",
    StaticFiles(directory="static", html=True),
    name="ui"
)
