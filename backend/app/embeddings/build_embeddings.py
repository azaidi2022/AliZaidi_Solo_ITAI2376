import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


MODEL_NAME = "all-MiniLM-L6-v2"

PROCESSED_DATA_PATH = Path("../data/processed/sistani_corpus.json")
VECTORSTORE_DIR = Path("../data/vectorstore")


def build_embeddings():
    print("Loading corpus...")

    with open(PROCESSED_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [item["text"] for item in data]

    print(f"Total chunks: {len(texts)}")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    embeddings = np.array(embeddings).astype("float32")

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(VECTORSTORE_DIR / "faiss_index.bin"))

    with open(VECTORSTORE_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(data, f)

    print("Embeddings + index saved successfully.")


if __name__ == "__main__":
    build_embeddings()