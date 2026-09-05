from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Week 3 Retrieval Service")

DOCUMENT_PATH = Path("../knowledge-base/documents/academic_policies.txt")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def load_document():
    if not DOCUMENT_PATH.exists():
        return ""

    return DOCUMENT_PATH.read_text(encoding="utf-8")


def chunk_document():
    content = load_document()

    if not content:
        return []

    sections = content.split("\n\n")

    chunks = []

    for section in sections:
        section = section.strip()

        if section:
            chunks.append(section)

    return chunks


def create_embeddings(chunks):
    return embedding_model.encode(chunks)


@app.get("/")
def root():
    return {
        "service": "Retrieval Service",
        "status": "running",
        "embedding_model": "all-MiniLM-L6-v2"
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


@app.get("/embeddings")
def get_embeddings():
    chunks = chunk_document()

    if not chunks:
        return {
            "error": "No chunks found"
        }

    vectors = create_embeddings(chunks)

    return {
        "number_of_chunks": len(chunks),
        "vector_dimensions": vectors.shape[1],
        "vectors": vectors.tolist()
    }


@app.get("/search")
def search(query: str):
    chunks = chunk_document()

    if not chunks:
        return {
            "error": "No chunks found"
        }

    chunk_vectors = create_embeddings(chunks)

    query_vector = embedding_model.encode([query])

    similarities = cosine_similarity(
        query_vector,
        chunk_vectors
    )[0]

    results = []

    for i, score in enumerate(similarities):
        results.append({
            "chunk": chunks[i],
            "score": float(score)
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    top_results = results[:2]

    return {
        "query": query,
        "relevant_context": top_results
    }
app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")
