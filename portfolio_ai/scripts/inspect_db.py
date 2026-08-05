import sys
from pathlib import Path

# Ensure portfolio_ai root directory is in sys.path
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.vectorstore.chroma_repository import ChromaRepository


def inspect_chroma_db():
    repo = ChromaRepository()
    collection = repo.collection
    count = collection.count()

    print("=" * 60)
    print(f"  CHROMA DB COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Collection Name : {collection.name}")
    print(f"Total Chunks    : {count}\n")

    if count == 0:
        print("No documents found in ChromaDB.")
        return

    # Fetch all records stored in ChromaDB
    all_docs = collection.get(include=["documents", "metadatas"])

    ids = all_docs.get("ids", [])
    documents = all_docs.get("documents", [])
    metadatas = all_docs.get("metadatas", [])

    print("=" * 60)
    print(f"  ALL STORED CHUNKS ({count} Total)")
    print("=" * 60)

    for i in range(count):
        doc_id = ids[i] if i < len(ids) else "N/A"
        meta = metadatas[i] if i < len(metadatas) else {}
        text = documents[i] if i < len(documents) else ""

        source = meta.get("source", "N/A")
        title = meta.get("title", "N/A")
        chunk_idx = meta.get("chunk", "N/A")

        # Preview first 120 characters of text
        snippet = text.replace("\n", " ")[:120] + ("..." if len(text) > 120 else "")

        print(f"[{i+1}/{count}] ID: {doc_id}")
        print(f"      Source   : {source} (Title: '{title}', Chunk: {chunk_idx})")
        print(f"      Snippet  : {snippet}\n")


if __name__ == "__main__":
    inspect_chroma_db()
