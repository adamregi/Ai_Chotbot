import sys
from pathlib import Path

# Ensure portfolio_ai root directory is in sys.path
PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import uuid
from langchain_chroma import Chroma

from config.settings import CHROMA_DB_DIR
from src.embeddings.embedding_service import (
    create_embedding,
    create_embeddings,
    get_embedding_model,
)


class ChromaRepository:
    """Persistent Chroma storage for portfolio document chunks using LangChain Chroma."""

    def __init__(self):
        self._embedding_function = get_embedding_model()
        self._vectorstore = Chroma(
            collection_name="portfolio",
            persist_directory=str(CHROMA_DB_DIR),
            embedding_function=self._embedding_function,
        )
        self.collection = getattr(self._vectorstore, "_collection", None)
        self.client = getattr(self._vectorstore, "_client", None)

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        if not texts:
            return

        doc_ids = ids if ids is not None else [str(uuid.uuid4()) for _ in texts]
        embeddings = create_embeddings(texts, input_type="passage")

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

        emb = create_embedding(question, input_type="query")
        results = self.collection.query(
            query_embeddings=[emb],
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

    def as_retriever(self, **kwargs):
        """Return a LangChain retriever backed by this vector store."""
        return self._vectorstore.as_retriever(**kwargs)
