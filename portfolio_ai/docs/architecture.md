# Architecture

```text
PDF -> readers/pdf_reader.py -> chunker.py -> embedding_service.py -> ChromaRepository
                                                                        |
HTTP request -> app.py -> chat_service.py -> rag_service.py -> BaseLLM
                                                          |             |
                                                          |       +-- NvidiaClient
                                                          |       +-- OllamaClient
```

`rag_service.py` retrieves portfolio context and builds a user prompt. It contains no provider-specific HTTP code. `BaseLLM` defines the generation contract, and `llm/factory.py` selects the client from `LLM_PROVIDER`.

`NvidiaClient` calls NVIDIA's OpenAI-compatible chat-completions endpoint using the existing `requests` dependency. Its key, base URL, model, timeout, completion-token limit, and retry behaviour come from `config/settings.py`, which loads `.env`. `OllamaClient` remains the local fallback.

Chroma stores portfolio knowledge only; it does not store conversation history. The generated database is local and Git-ignored, while `data/raw/resume.pdf` is the source used to rebuild it.
