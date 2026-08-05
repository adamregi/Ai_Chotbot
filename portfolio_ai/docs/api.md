# API

## `GET /`

Returns a service status message.

## `POST /chat`

Request body:

```json
{ "prompt": "Tell me about the portfolio" }
```

Successful response:

```json
{ "question": "Tell me about the portfolio", "answer": "..." }
```

`prompt` must contain 1–4,000 characters. Invalid input returns `422`; an unavailable Ollama service returns `503` with a diagnostic message.

## `GET /chat?prompt=...`

Legacy equivalent of `POST /chat`, retained for simple browser testing.
