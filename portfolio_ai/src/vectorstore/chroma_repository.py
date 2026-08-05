import uuid

import chromadb

from config.settings import CHROMA_DB_DIR
from src.embeddings.embedding_service import create_embedding


class ChromaRepository:
    """Persistent Chroma storage for portfolio document chunks."""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        self.collection = self.client.get_or_create_collection(name="portfolio")

    def add_documents(self, texts: list[str]) -> None:
        texts = [text.strip() for text in texts if text and text.strip()]
        if not texts:
            return
        self.collection.add(
            documents=texts,
            embeddings=[create_embedding(text) for text in texts],
            ids=[str(uuid.uuid4()) for _ in texts],
        )

    def search(self, question: str, top_k: int = 5) -> list[dict[str, str | float]]:
        if not question or not question.strip() or top_k <= 0:
            return []
        document_count = self.collection.count()
        if document_count == 0:
            return []
        results = self.collection.query(
            query_embeddings=[create_embedding(question)],
            n_results=min(top_k, document_count),
        )
        documents = results.get("documents", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []
        return [
            {"text": document, "distance": distance}
            for document, distance in zip(documents, distances)
        ]
