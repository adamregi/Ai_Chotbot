# Portfolio AI

A FastAPI RAG chatbot that retrieves portfolio context from Chroma and generates answers through NVIDIA AI or a local Ollama fallback.

## Setup

```powershell
cd .\portfolio_ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then add your NVIDIA API key:

```dotenv
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_MODEL=meta/llama-3.1-70b-instruct
# Leave false unless your network requires a working HTTP(S) proxy.
NVIDIA_USE_ENV_PROXY=false
```

The NVIDIA client uses the OpenAI-compatible `https://integrate.api.nvidia.com/v1/chat/completions` endpoint directly through `requests`; no OpenAI SDK is required. Use `LLM_PROVIDER=ollama` to run generation locally instead.

## Build the knowledge index

Put the latest resume at `data/raw/resume.pdf`, start Ollama for embeddings, then run:

```powershell
python scripts/rebuild_db.py
```

## Run and test

```powershell
uvicorn app:app --reload
pytest
```

In a second terminal, from the repository root, start the React app:

```powershell
npm run dev
```

The Vite development server forwards browser requests from `/api/*` to FastAPI
at `http://127.0.0.1:8000/*`. For a deployed frontend, set
`VITE_CHAT_API_URL` to the public FastAPI base URL before building.

Send `POST http://127.0.0.1:8000/chat` with:

```json
{ "prompt": "What projects has Vignesh built?" }
```

See [architecture.md](docs/architecture.md) and [api.md](docs/api.md).
