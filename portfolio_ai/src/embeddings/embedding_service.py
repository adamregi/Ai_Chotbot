import os
import requests

API_KEY = os.getenv("NVIDIA_API_KEY")
BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
MODEL = os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")


def create_embedding(text: str):
    payload = {
        "input": text,
        "model": MODEL,
        "input_type": "query"
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    print(f"[Embedding Request] POST {BASE_URL}/embeddings | Model: '{MODEL}'")
    
    response = requests.post(
        f"{BASE_URL}/embeddings",
        headers=headers,
        json=payload,
        timeout=60,
    )

    print("Status:", response.status_code)
    print("Response:", response.text)

    response.raise_for_status()

    return response.json()["data"][0]["embedding"]
