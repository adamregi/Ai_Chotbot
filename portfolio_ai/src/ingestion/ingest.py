import sys
from pathlib import Path

# Ensure portfolio_ai root directory is in sys.path
PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config.settings import RAW_DATA_DIR
from src.ingestion.chunker import chunk_text
from src.vectorstore.chroma_repository import ChromaRepository

EXCLUDED_FILES = {"system_context.md"}


def ingest(raw_dir: Path = RAW_DATA_DIR) -> tuple[int, int]:
    """Reads all raw Markdown and PDF files, chunks them, embeds them, and stores them in ChromaDB with metadata."""
    store = ChromaRepository()
    try:
        store.client.delete_collection("portfolio")
    except Exception:
        pass
    store = ChromaRepository()

    # Collect both .md and .pdf files
    data_files = sorted(
        [f for f in raw_dir.iterdir() if f.is_file() and f.suffix.lower() in {".md", ".pdf"}]
    )
    
    all_texts = []
    all_metadatas = []
    all_ids = []

    files_processed = 0

    print(f"Starting ingestion from: {raw_dir}\n")

    for file in data_files:
        if file.name.lower() in EXCLUDED_FILES:
            print(f"Skipping {file.name} (excluded from vector store)")
            continue

        text = ""
        if file.suffix.lower() == ".md":
            text = file.read_text(encoding="utf-8").strip()
        elif file.suffix.lower() == ".pdf":
            try:
                from src.ingestion.readers.pdf_reader import extract_pages
                pages = extract_pages(str(file))
                text = "\n\n".join([f"Page {p['page']}:\n{p['text']}" for p in pages]).strip()
            except Exception as e:
                print(f"Error reading PDF {file.name}: {e}")
                continue

        if not text:
            print(f"Skipping {file.name} (file is empty)")
            continue

        chunks = chunk_text(text)
        print(f"Reading {file.name} | Chunks: {len(chunks)}")

        category = file.stem
        title = file.stem.replace("_", " ").title()

        for i, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_metadatas.append({
                "source": file.name,
                "title": title,
                "category": category,
                "chunk": i,
            })
            all_ids.append(f"{file.stem}_{i}")

        files_processed += 1

    if all_texts:
        store.add_documents(texts=all_texts, metadatas=all_metadatas, ids=all_ids)

    print("\nFinished.")
    print(f"{files_processed} files processed")
    print(f"{len(all_texts)} chunks stored in ChromaDB.")

    return files_processed, len(all_texts)



if __name__ == "__main__":
    ingest()
