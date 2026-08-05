from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config.settings import CORS_ORIGINS, LLM_PROVIDER, NVIDIA_MODEL, OLLAMA_CHAT_MODEL, EMBEDDING_MODEL
from src.schemas import ChatRequest, ChatResponse
from src.services.chat_service import ask_portfolio

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_answer(prompt: str) -> ChatResponse:
    try:
        return ChatResponse(question=prompt, answer=ask_portfolio(prompt))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/")
def home() -> dict[str, str]:
    return {"message": "Portfolio Chatbot"}


@app.get("/health")
def health() -> dict[str, str]:
    model = NVIDIA_MODEL if LLM_PROVIDER == "nvidia" else OLLAMA_CHAT_MODEL
    return {
        "status": "ok",
        "provider": LLM_PROVIDER,
        "model": model,
    }


@app.get("/info")
def info() -> dict[str, str]:
    model = NVIDIA_MODEL if LLM_PROVIDER == "nvidia" else OLLAMA_CHAT_MODEL
    return {
        "assistant": "Vignesh AI",
        "version": "1.0",
        "provider": LLM_PROVIDER,
        "model": model,
        "embedding": EMBEDDING_MODEL,
        "vectorStore": "ChromaDB",
    }



@app.get("/chat", response_model=ChatResponse)
def chat(prompt: str) -> ChatResponse:
    """Legacy endpoint; new clients should send POST /chat."""
    return get_answer(prompt)


@app.post("/chat", response_model=ChatResponse)
def chat_post(request: ChatRequest) -> ChatResponse:
    return get_answer(request.prompt)
