import sys
from pathlib import Path

# Add embeddings folder to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "embeddings"))

# Add models folder to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))

from query_embeddings import search
from cnn_page_classifier import CNNPageClassifier
from text_to_tensor import text_to_tensor

import torch


def plan(question: str) -> dict:
    """
    PLAN step:
    Decide what the agent needs to do.
    """

    return {
        "question": question,
        "steps": [
            "Understand the user's fiqh question",
            "Retrieve relevant rulings from the Sistani corpus",
            "Use retrieved evidence to form a cautious answer",
            "Return answer with page references"
        ]
    }


def act(question: str) -> list[dict]:
    """
    ACT step:
    Use the retrieval tool.
    """

    retrieved_chunks = search(question, top_k=5)
    return retrieved_chunks


def observe(retrieved_chunks: list[dict]) -> str:
    """
    OBSERVE step:
    Format retrieved evidence for reasoning.
    """

    observations = []

    for chunk in retrieved_chunks:
        observations.append(
            f"Source: {chunk['source_name']} | Scholar: {chunk['scholar']} | "
            f"Page: {chunk['page_number']}\n"
            f"Text: {chunk['text']}"
        )

    return "\n\n---\n\n".join(observations)


def respond(question: str, retrieved_chunks: list[dict]) -> dict:
    best_chunk = retrieved_chunks[0]
    evidence_text = best_chunk["text"][:1400]

    q = question.lower()
    e = evidence_text.lower()

    if "qasr" in q or "short" in q or "traveller" in q or "travel" in q or "tarakhkhus" in q:
        short_answer = (
            "Based on the retrieved rulings, a traveller prays qaṣr when the journey meets the required travel distance "
            "of eight farsakhs or more and the person has passed the tarakhkhus limit. The tarakhkhus limit means the point "
            "outside the town/city where the traveller is no longer considered locally present, such as when the signs of the city "
            "are no longer normally seen or the adhān is no longer normally heard. In simple terms: once the trip qualifies as a "
            "sharʿī journey and the traveller has left the local boundary, the prayer becomes shortened."
    )
    elif "najis" in q or "impure" in q or "alcohol" in q or "wine" in q:
        short_answer = (
            "According to the retrieved ruling, wine is najis/impure. However, alcohol in general "
            "is not automatically najis; industrial or medicinal alcohol is pure unless it is known "
            "to come from the vaporisation and distillation of grape wine."
        )
    elif "wudhu" in q or "wuḍū" in q or "ablution" in q:
        short_answer = (
            "The retrieved ruling relates to wuḍūʾ/ablution. Based on the evidence, the answer "
            "should be understood through the specific ruling text shown below."
        )
    else:
        short_answer = (
            "Based on the most relevant retrieved ruling, the answer depends on the specific "
            "conditions mentioned in the ruling text below."
        )

    return {
        "question": question,
        "answer": {
            "short_answer": short_answer,
            "evidence": evidence_text,
            "reference": {
                "book": best_chunk["source_name"],
                "scholar": best_chunk["scholar"],
                "page": best_chunk["page_number"],
                "chunk": best_chunk["chunk_id"]
            },
            "note": "This is an AI-assisted explanation based on retrieved evidence. Verify personal rulings with the official source or a qualified scholar."
        },
        "sources": [
            {
                "source_name": chunk["source_name"],
                "scholar": chunk["scholar"],
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"]
            }
            for chunk in retrieved_chunks
        ]
    }


def classify_relevance(chunks: list[dict]) -> list[dict]:
    """
    Tool 2:
    CNN-based page/chunk relevance filter.

    Note: this is currently an untrained CNN prototype,
    so we use it as a demonstration tool and keep fallback behavior.
    """

    model = CNNPageClassifier()
    model.eval()

    filtered = []

    with torch.no_grad():
        for chunk in chunks:
            text = chunk["text"][:500]
            tensor = text_to_tensor(text)

            output = model(tensor)
            prediction = torch.argmax(output, dim=1).item()

            chunk["cnn_prediction"] = prediction

            if prediction == 0:
                filtered.append(chunk)

    if not filtered:
        return chunks

    return filtered


def run_agent(question: str) -> dict:
    """
    Full agent loop:
    Plan → Act → Observe → Respond
    """

    agent_plan = plan(question)
    retrieved_chunks = act(question)
    observation = observe(retrieved_chunks)
    final_response = respond(question, retrieved_chunks)

    return {
        "plan": agent_plan,
        "observation": observation,
        "response": final_response
    }


if __name__ == "__main__":
    result = run_agent("What inalidates Wudhu?")

    print("\nPLAN:")
    print(result["plan"])

    print("\nOBSERVATION:")
    print(result["observation"][:1500])

    print("\nRESPONSE:")
    print(result["response"])