import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).resolve().parents[1] / "data" / "vectorstore"

print("Loading embedding model once...")
MODEL = SentenceTransformer(MODEL_NAME)
print("Model loaded.")


def load_index_and_metadata():
    index = faiss.read_index(str(VECTORSTORE_DIR / "faiss_index.bin"))

    with open(VECTORSTORE_DIR / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


def keyword_score(query: str, text: str) -> int:
    """
    Simple keyword reranker to improve retrieval accuracy.
    """

    query = query.lower()
    text = text.lower()

    score = 0

    important_terms = [
        "najis",
        "impure",
        "pure",
        "wine",
        "alcohol",
        "intoxicat",
        "beer",
        "fuqqā",
        "fuqqa"
    ]

    for term in important_terms:
        if term in query and term in text:
            score += 5
        elif term in text:
            score += 1

    if "ruling" in text.lower():
        score += 1

    return score


def search(query: str, top_k: int = 5, candidate_k: int = 15):
    print(f"\nQuery: {query}")

    index, metadata = load_index_and_metadata()

    query_embedding = MODEL.encode([query]).astype("float32")

    distances, indices = index.search(query_embedding, candidate_k)

    candidates = []

    for rank, i in enumerate(indices[0]):
        item = metadata[i]
        item = item.copy()

        item["faiss_rank"] = rank + 1
        item["keyword_score"] = keyword_score(query, item["text"])

        candidates.append(item)

    reranked = sorted(
        candidates,
        key=lambda x: (x["keyword_score"], -x["faiss_rank"]),
        reverse=True
    )

    return reranked[:top_k]


if __name__ == "__main__":
    results = search("Is alcohol najis?")

    for i, res in enumerate(results):
        print("\n" + "=" * 50)
        print(f"Result {i + 1}")
        print(f"Page: {res['page_number']}")
        print(f"FAISS rank: {res['faiss_rank']}")
        print(f"Keyword score: {res['keyword_score']}")
        print(res["text"][:700])