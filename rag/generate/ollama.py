"""Call the loopback Ollama generation endpoint."""

import httpx

from rag.config import LLM
from rag.generate.prompt import SYSTEM


class OllamaError(RuntimeError):
    """A failed request or invalid response from the local Ollama server."""


def generate(
    prompt: str,
    settings: LLM,
    *,
    system: str = SYSTEM,
    output_schema: dict[str, object] | None = None,
) -> str:
    """Generate a single answer and release the model after the response."""
    request = {
        "model": settings.model_name,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "keep_alive": 0,
        "think": settings.think,
        "options": {
            "temperature": settings.temperature,
            "num_predict": settings.max_tokens,
            "num_ctx": settings.context_window,
            "seed": settings.seed,
        },
    }
    if output_schema is not None:
        request["format"] = output_schema
    try:
        with httpx.Client(timeout=settings.timeout_seconds, trust_env=False) as client:
            response = client.post(
                f"{str(settings.base_url).rstrip('/')}/api/generate", json=request
            )
            response.raise_for_status()
            answer = response.json()["response"]
            if not isinstance(answer, str):
                raise TypeError("Ollama response field must be text")
            return answer
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
        raise OllamaError(f"Ollama generation failed: {error}") from error
