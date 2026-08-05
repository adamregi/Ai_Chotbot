from config.settings import RESUME_PATH
from src.ingestion.chunker import chunk_text
from src.ingestion.readers.pdf_reader import extract_pages
from src.vectorstore.chroma_repository import ChromaRepository


def ingest(pdf_path=RESUME_PATH) -> int:
    """Replace the persistent portfolio index with chunks from a PDF."""
    store = ChromaRepository()
    try:
        store.client.delete_collection("portfolio")
    except ValueError:
        pass
    store = ChromaRepository()

    chunks = []
    for page in extract_pages(str(pdf_path)):
        chunks.extend(chunk_text(str(page["text"])))
    store.add_documents(chunks)
    return len(chunks)


if __name__ == "__main__":
    print(f"Portfolio ingested successfully ({ingest()} chunks).")
