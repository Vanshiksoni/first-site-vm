from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Retrieval / Knowledge Base Service")

DOCUMENTS = [
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

chunks = [
    f"{doc['title']}\n{doc['content']}"
    for doc in DOCUMENTS
]

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

vectors = vectorizer.fit_transform(chunks)


@app.get("/")
def root():
    return {
        "service": "Retrieval / Knowledge Base Service",
        "status": "running"
    }


@app.get("/search")
def search(query: str):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, vectors)[0]

    ranked = []

    for doc, score in zip(DOCUMENTS, scores):
        title = doc["title"].lower()
        query_lower = query.lower()

        if "assignment" in query_lower and "assignment" in title:
            score += 1.0
        elif "attendance" in query_lower and "attendance" in title:
            score += 1.0
        elif "examination" in query_lower and "examination" in title:
            score += 1.0
        elif "leave" in query_lower and "leave" in title:
            score += 1.0
        elif "grading" in query_lower and "grading" in title:
            score += 1.0

        ranked.append((doc, score))

    ranked.sort(key=lambda x: x[1], reverse=True)

    results = []

    for doc, score in ranked[:2]:
        results.append({
            "chunk": f"{doc['title']}\n{doc['content']}",
            "score": float(score)
        })

    return {
        "query": query,
        "relevant_context": results
    }


app.mount(
    "/ui",
    StaticFiles(directory="static", html=True),
    name="ui"
)
