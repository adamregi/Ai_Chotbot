import os
import requests

API_KEY = os.getenv("NVIDIA_API_KEY")
BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
MODEL = os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")


def create_embeddings(texts: list[str], input_type: str = "passage") -> list[list[float]]:
    if not texts:
        return []

    # Handle single string or list of strings
    payload = {
        "input": texts,
        "model": MODEL,
        "input_type": input_type,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        f"{BASE_URL}/embeddings",
        headers=headers,
        json=payload,
        timeout=60,
    )

    response.raise_for_status()
    res_data = response.json()["data"]
    
    # Sort embeddings by original input index
    sorted_items = sorted(res_data, key=lambda x: x.get("index", 0))
    return [item["embedding"] for item in sorted_items]


def create_embedding(text: str, input_type: str = "query") -> list[float]:
    embeddings = create_embeddings([text], input_type=input_type)
    return embeddings[0] if embeddings else []


