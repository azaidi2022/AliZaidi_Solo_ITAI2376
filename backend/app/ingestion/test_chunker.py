from pdf_loader import load_pdf_pages
from chunker import clean_text, chunk_text
from metadata import create_chunk_record


pages = load_pdf_pages(
    pdf_path="../data/raw/sistani_islamic_laws_2023.pdf",
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

print(f"Total chunks created: {len(all_chunks)}")

print("\nSample chunk:")
print(all_chunks[0])