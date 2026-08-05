import time

import requests
from requests import RequestException

from config.settings import (
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_MAX_COMPLETION_TOKENS,
    NVIDIA_MAX_RETRIES,
    NVIDIA_MODEL,
    NVIDIA_RETRY_DELAY_SECONDS,
    NVIDIA_USE_ENV_PROXY,
    REQUEST_TIMEOUT_SECONDS,
)
from src.llm.base_llm import BaseLLM


class NvidiaClient(BaseLLM):
    """NVIDIA AI client using the OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        api_key: str | None = NVIDIA_API_KEY,
        base_url: str = NVIDIA_BASE_URL,
        model: str = NVIDIA_MODEL,
        max_completion_tokens: int = NVIDIA_MAX_COMPLETION_TOKENS,
        max_retries: int = NVIDIA_MAX_RETRIES,
        retry_delay_seconds: float = NVIDIA_RETRY_DELAY_SECONDS,
        use_env_proxy: bool = NVIDIA_USE_ENV_PROXY,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.session = requests.Session()
        # Ignore broken inherited proxy variables unless the deployment explicitly
        # opts into them with NVIDIA_USE_ENV_PROXY=true.
        self.session.trust_env = use_env_proxy

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY is not configured. Add it to .env; never place it in source code."
            )
        if self.max_completion_tokens <= 0:
            raise RuntimeError("NVIDIA_MAX_COMPLETION_TOKENS must be greater than zero.")

        payload = {
            "model": self.model,
            "max_tokens": self.max_completion_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            except RequestException as exc:
                if attempt == self.max_retries:
                    proxy_hint = (
                        " The configured system proxy could not be reached; set "
                        "NVIDIA_USE_ENV_PROXY=false or fix the proxy."
                        if "ProxyError" in type(exc).__name__ or "proxy" in str(exc).lower()
                        else ""
                    )
                    raise RuntimeError(
                        "Could not reach the NVIDIA API. Check your network connection and "
                        f"NVIDIA_BASE_URL.{proxy_hint}"
                    ) from exc
                time.sleep(self.retry_delay_seconds * (2**attempt))
                continue

            if response.status_code == 429:
                if attempt == self.max_retries:
                    raise RuntimeError("NVIDIA rate limit reached. Wait briefly or check your API plan.")
                time.sleep(self.retry_delay_seconds * (2**attempt))
                continue

            try:
                response.raise_for_status()
            except RequestException as exc:
                raise RuntimeError(
                    f"NVIDIA request failed (HTTP {response.status_code}). Check NVIDIA_API_KEY and NVIDIA_MODEL."
                ) from exc

            try:
                answer = response.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise RuntimeError("NVIDIA returned an invalid chat-completion response.") from exc

            answer = (answer or "").strip()
            if not answer:
                raise RuntimeError("NVIDIA returned an empty response.")
            return answer

        raise RuntimeError("NVIDIA request failed after retries.")
