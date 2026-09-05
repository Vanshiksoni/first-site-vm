from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests

app = FastAPI(title="University Student Helpdesk")

RETRIEVAL_URL = "http://week3-retrieval:8001/search"
LLM_URL = "http://week3-llm:8000/generate"

class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "service": "Application / Orchestration Service",
        "status": "running"
    }


@app.post("/ask")
def ask(request: QuestionRequest):

    # Step 1: Retrieve relevant knowledge
    retrieval_response = requests.get(
        RETRIEVAL_URL,
        params={"query": request.question},
        timeout=30
    )
    retrieval_response.raise_for_status()

    retrieval_data = retrieval_response.json()

    relevant_context = "\n\n".join(
        item["chunk"]
        for item in retrieval_data["relevant_context"]
    )

    # Step 2: Send question and context to LLM
    try:
        llm_response = requests.post(
            LLM_URL,
            json={
                "prompt": request.question,
                "context": relevant_context
            },
            timeout=120
        )

        llm_response.raise_for_status()
        llm_data = llm_response.json()

        answer = llm_data["response"]
        model = llm_data["model"]
        llm_used = True

    except Exception:
        # Lightweight fallback if Ollama is unavailable
        answer = (
            "Based on the university knowledge base:\n\n"
            + relevant_context
        )
        model = "Code Llama (fallback demonstration)"
        llm_used = False

    return {
        "question": request.question,
        "answer": answer,
        "rag_used": True,
        "retrieved_context": retrieval_data["relevant_context"],
        "model": model,
        "llm_used": llm_used
    }


app.mount(
    "/ui",
    StaticFiles(directory="static", html=True),
    name="ui"
)
