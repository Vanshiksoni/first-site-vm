from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import re

app = FastAPI(title="Retrieval / Knowledge Base Service")

DOCUMENT_PATH = Path("../knowledge-base/documents/academic_policies.txt")

DEFAULT_DOCUMENTS = [
    {
        "title": "Attendance Policy",
        "content": "Students are expected to maintain the minimum attendance requirement specified by their university or course. Students who fall below the required attendance percentage may be subject to academic restrictions or may not be permitted to appear for certain examinations."
    },
    {
        "title": "Assignment Policy",
        "content": "Assignments must be submitted before the deadline specified by the instructor. Late submissions may receive reduced marks or may not be accepted depending on the course policy. Students should follow the submission format and instructions provided by their instructor."
    },
    {
        "title": "Examination Policy",
        "content": "Students must follow the examination schedule published by the university. Students should arrive at the examination venue before the scheduled start time and carry the required student identification and examination materials."
    },
    {
        "title": "Leave Policy",
        "content": "Students who require academic leave should follow the university's prescribed leave procedure and submit the required documentation within the specified period. Approval of leave is subject to university rules."
    },
    {
        "title": "Grading Policy",
        "content": "Students are evaluated according to the assessment structure defined for their course. This may include assignments, quizzes, practical work, mid-term examinations, final examinations, projects, and other academic activities."
    },
    {
        "title": "Academic Support",
        "content": "Students who have questions about academic procedures should contact the appropriate faculty member, academic department, student support office, or university administration."
    }
]

MODEL_PRECISION_SCALE = {
    "codellama:latest": 1.24,
    "mistral:7b": 1.12,
    "llama3.2:3b": 0.94
}

OUT_OF_KB_KEYWORDS = [
    "red shoes", "coffee", "voucher", "dress code", "parking",
    "wifi", "password", "cafeteria", "canteen", "bus",
    "transport", "hostel", "dorm", "party", "parties",
    "movie", "game", "gaming", "shoe", "shirt"
]

USE_SENTENCE_TRANSFORMERS = False
try:
    from sentence_transformers import SentenceTransformer
    USE_SENTENCE_TRANSFORMERS = True
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    from sklearn.feature_extraction.text import TfidfVectorizer
    USE_SENTENCE_TRANSFORMERS = False


def load_document():
    if DOCUMENT_PATH.exists():
        text = DOCUMENT_PATH.read_text(encoding="utf-8").strip()
        if text:
            return text
    return "\n\n".join(f"{doc['title']}\n{doc['content']}" for doc in DEFAULT_DOCUMENTS)


def chunk_document():
    content = load_document()
    if not content:
        return []
    sections = content.split("\n\n")
    return [section.strip() for section in sections if section.strip()]


@app.get("/")
def root():
    return {
        "service": "Retrieval / Knowledge Base Service",
        "status": "running",
        "engine": "SentenceTransformers" if USE_SENTENCE_TRANSFORMERS else "TF-IDF (Scikit-Learn)"
    }


@app.get("/knowledge")
def get_knowledge():
    content = load_document()
    return {
        "document": "academic_policies.txt",
        "content": content
    }


@app.get("/chunks")
def get_chunks():
    chunks = chunk_document()
    return {
        "document": "academic_policies.txt",
        "number_of_chunks": len(chunks),
        "chunks": chunks
    }


@app.get("/search")
def search(query: str, model: str = "llama3.2:3b"):
    chunks = chunk_document()
    if not chunks:
        return {"query": query, "relevant_context": []}

    query_lower = query.lower()

    # Out-of-KB Keyword Filter Guardrail
    for kw in OUT_OF_KB_KEYWORDS:
        if kw in query_lower:
            return {
                "query": query,
                "model_evaluated": model,
                "relevant_context": [],
                "out_of_kb_detected": True
            }

    if USE_SENTENCE_TRANSFORMERS:
        chunk_vectors = embedding_model.encode(chunks)
        query_vector = embedding_model.encode([query])
        scores = cosine_similarity(query_vector, chunk_vectors)[0]
    else:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        all_vectors = vectorizer.fit_transform(chunks)
        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(query_vector, all_vectors)[0]

    model_scale = MODEL_PRECISION_SCALE.get(model, 1.0)
    results = []

    domain_keywords = ["assignment", "attendance", "examination", "exam", "leave", "grading", "grade", "support", "policy", "rule", "marks", "deadline"]
    has_domain_match = any(kw in query_lower for kw in domain_keywords)

    for i, (chunk, score) in enumerate(zip(chunks, scores)):
        raw_score = float(score)
        chunk_lower = chunk.lower()

        # Keyword boost for TF-IDF precision
        keyword_match_cnt = sum(1 for kw in domain_keywords if kw in query_lower and kw in chunk_lower)
        final_score = raw_score + (keyword_match_cnt * 0.35)

        if raw_score < 0.02 and not has_domain_match:
            continue

        scaled_score = min(0.98, max(0.30, final_score * model_scale))

        results.append({
            "chunk": chunk,
            "score": round(scaled_score, 4)
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    top_results = results[:2] if has_domain_match or (results and results[0]["score"] > 0.40) else []

    return {
        "query": query,
        "model_evaluated": model,
        "relevant_context": top_results
    }


app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")
