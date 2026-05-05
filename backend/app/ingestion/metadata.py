def create_chunk_record(
    chunk_text: str,
    source_name: str,
    scholar: str,
    page_number: int,
    chunk_id: int
) -> dict:
    """
    Create one searchable chunk with citation metadata.
    """

    return {
        "id": f"{scholar.lower()}_page_{page_number}_chunk_{chunk_id}",
        "text": chunk_text,
        "source_name": source_name,
        "scholar": scholar,
        "page_number": page_number,
        "chunk_id": chunk_id
    }