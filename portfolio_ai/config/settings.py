from pathlib import Path
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
RESUME_PATH = RAW_DATA_DIR / "resume.pdf"

# FastEmbed local embeddings model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Answer-generation provider: 'nvidia' or 'ollama'.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "nvidia").strip().lower()

# NVIDIA AI uses an OpenAI-compatible HTTP endpoint.
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.2-11b-vision-instruct")
NVIDIA_MAX_COMPLETION_TOKENS = int(os.getenv("NVIDIA_MAX_COMPLETION_TOKENS", "1024"))
NVIDIA_MAX_RETRIES = int(os.getenv("NVIDIA_MAX_RETRIES", "2"))
NVIDIA_RETRY_DELAY_SECONDS = float(os.getenv("NVIDIA_RETRY_DELAY_SECONDS", "1"))
# Requests otherwise inherits HTTP(S)_PROXY values from the operating system. Keep
# that opt-in: a stale local proxy should not prevent a hosted NVIDIA request.
NVIDIA_USE_ENV_PROXY = os.getenv("NVIDIA_USE_ENV_PROXY", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Ollama is a local chat fallback.
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]
