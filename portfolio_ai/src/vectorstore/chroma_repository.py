import sys
from pathlib import Path

# Ensure portfolio_ai root directory is in sys.path
PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import uuid
import chromadb

from config.settings import CHROMA_DB_DIR
from src.embeddings.embedding_service import create_embedding


class ChromaRepository:
    """Persistent Chroma storage for portfolio document chunks."""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        self.collection = self.client.get_or_create_collection(name="portfolio")

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        if not texts:
            return
        
        doc_ids = ids if ids is not None else [str(uuid.uuid4()) for _ in texts]
        embeddings = [create_embedding(text) for text in texts]

        kwargs = {
            "documents": texts,
            "embeddings": embeddings,
            "ids": doc_ids,
        }
        if metadatas is not None:
            kwargs["metadatas"] = metadatas

        self.collection.add(**kwargs)

    def search(self, question: str, top_k: int = 5) -> list[dict]:
        if not question or not question.strip() or top_k <= 0:
            return []
        document_count = self.collection.count()
        if document_count == 0:
            return []
        results = self.collection.query(
            query_embeddings=[create_embedding(question)],
            n_results=min(top_k, document_count),
            include=["documents", "metadatas", "distances"],
        )
        documents = results.get("documents", [[]])[0] or []
        metadatas = results.get("metadatas", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []

        output = []
        for i, document in enumerate(documents):
            meta = metadatas[i] if i < len(metadatas) else {}
            dist = distances[i] if i < len(distances) else 0.0
            output.append({
                "text": document,
                "metadata": meta,
                "distance": dist,
            })
        return output
