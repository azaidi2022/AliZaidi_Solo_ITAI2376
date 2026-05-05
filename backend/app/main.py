from fastapi import FastAPI
from pydantic import BaseModel

from app.agent.ahkam_agent import run_agent


app = FastAPI(
    title="Ahkam Navigator API",
    description="A Shia rulings agent using RAG, SentenceTransformers, and a CNN document-processing model.",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "Ahkam Navigator API is running",
        "status": "ok"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    from app.agent.ahkam_agent import run_agent

    result = run_agent(request.question)

    return {
        "question": request.question,
        "answer": result["response"]["answer"],
        "sources": result["response"]["sources"],
        "agent_loop": {
            "plan": result["plan"]["steps"],
            "act": "Retrieved relevant chunks using SentenceTransformer embeddings and FAISS.",
            "observe": "Reviewed retrieved evidence from the Sistani corpus.",
            "respond": "Returned a structured answer with source reference."
        }
    }