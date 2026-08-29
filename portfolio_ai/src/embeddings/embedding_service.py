import os
from typing import List, Union
from fastembed import TextEmbedding

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
_model_instance = None


def get_embedding_model() -> TextEmbedding:
    """Lazy-load the FastEmbed TextEmbedding model."""
    global _model_instance
    if _model_instance is None:
        _model_instance = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    return _model_instance


def create_embeddings(texts: Union[str, List[str]], input_type: str = "passage") -> List[List[float]]:
    """
    Generate vector embeddings using FastEmbed (local, ONNX-accelerated).
    
    Args:
        texts: A single text string or list of text strings.
        input_type: Retained for signature compatibility ('passage' or 'query').
    """
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return []

    model = get_embedding_model()
    embeddings_iter = model.embed(texts)
    return [e.tolist() for e in embeddings_iter]


def create_embedding(text: str, input_type: str = "query") -> List[float]:
    """Generate embedding for a single text."""
    embeddings = create_embeddings([text], input_type=input_type)
    return embeddings[0] if embeddings else []



