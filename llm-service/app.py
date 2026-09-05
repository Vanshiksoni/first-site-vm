from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests

app = FastAPI(title="Week 3 LLM Service")

OLLAMA_URL = "http://host.docker.internal:11434/api/generate"
MODEL = "codellama"


class GenerateRequest(BaseModel):
    prompt: str
    context: str = ""


@app.get("/")
def root():
    return {
        "service": "LLM Service",
        "status": "running",
        "model": MODEL
    }


@app.post("/generate")
def generate(request: GenerateRequest):

    if request.context:
        final_prompt = f"""
Use the following university knowledge base context to answer the question.

KNOWLEDGE BASE CONTEXT:
{request.context}

QUESTION:
{request.prompt}

Answer the question using the provided context. If the context does not contain enough information, say that the information is not available in the knowledge base.
"""
    else:
        final_prompt = request.prompt

    payload = {
        "model": MODEL,
        "prompt": final_prompt,
        "stream": False,
        "options": {
            "num_ctx": 256,
            "num_predict":100 
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    result = response.json()

    return {
        "response": result.get("response", ""),
        "model": result.get("model", MODEL),
        "rag_used": bool(request.context)
    }


app.mount(
    "/ui",
    StaticFiles(directory="static", html=True),
    name="ui"
)
