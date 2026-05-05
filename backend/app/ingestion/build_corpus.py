import json
from pathlib import Path

from pdf_loader import load_pdf_pages
from chunker import clean_text, chunk_text
from metadata import create_chunk_record


RAW_DATA_DIR = Path("../data/raw")
PROCESSED_DATA_DIR = Path("../data/processed")


def build_corpus():
    pages = load_pdf_pages(
        pdf_path=RAW_DATA_DIR / "sistani_islamic_laws_2023.pdf",
        source_name="Islamic Laws 2023",
        scholar="Sistani"
    )

    all_chunks = []

    for page in pages:
        cleaned_text = clean_text(page["text"])
        chunks = chunk_text(cleaned_text)

        for i, chunk in enumerate(chunks):
            record = create_chunk_record(
                chunk_text=chunk,
                source_name=page["source_name"],
                scholar=page["scholar"],
                page_number=page["page_number"],
                chunk_id=i
            )

            all_chunks.append(record)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DATA_DIR / "sistani_corpus.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_chunks)} chunks to {output_path}")


if __name__ == "__main__":
    build_corpus()