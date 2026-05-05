from pdf_loader import load_pdf_pages


pages = load_pdf_pages(
    pdf_path="../data/raw/sistani_islamic_laws_2023.pdf",
    source_name="Islamic Laws 2023",
    scholar="Sistani"
)

print(f"Total pages: {len(pages)}")

for page in pages:
    if page["text"].strip():
        print("\nFIRST READABLE PAGE FOUND")
        print(f"Page: {page['page_number']}")
        print(page["text"][:1500])
        break
else:
    print("No readable text found.")