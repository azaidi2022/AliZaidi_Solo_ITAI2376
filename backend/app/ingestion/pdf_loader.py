from pathlib import Path
import fitz  # PyMuPDF


def load_pdf_pages(pdf_path: str, source_name: str, scholar: str) -> list[dict]:
    """
    Load a PDF page by page and extract readable text.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = fitz.open(str(pdf_path))
    print(f"PDF opened successfully. Pages found: {len(document)}")

    pages = []

    for page_index in range(len(document)):
        page = document[page_index]
        text = page.get_text("text") or ""

        pages.append({
            "source_name": source_name,
            "scholar": scholar,
            "page_number": page_index + 1,
            "text": text.strip()
        })

    document.close()
    return pages