"""LLM provider abstraction. Gemini is the first provider; add another by subclassing LLMProvider and
calling register_provider(). Keys are read from the environment only."""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Callable

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMProvider(ABC):
    name = "llm"
    model = ""

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.2, json_mode: bool = False) -> str: ...


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", max_retries: int = 4):
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        self.model, self.max_retries, self._api_key, self._client = model, max_retries, api_key, None

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def generate(self, prompt, system=None, temperature=0.2, json_mode=False) -> str:
        from google.genai import types

        cfg = types.GenerateContentConfig(system_instruction=system, temperature=temperature,
                                          response_mime_type="application/json" if json_mode else None)
        delay = 2.0
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._get_client().models.generate_content(model=self.model, contents=prompt, config=cfg)
                if not resp.text:
                    raise LLMError("The model returned an empty response (possibly blocked by safety filters)")
                return resp.text
            except LLMError:
                raise
            except Exception as exc:
                msg = str(exc)
                transient = any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "timeout"))
                if transient and attempt < self.max_retries:
                    log.warning("Gemini transient error (%s); retry %d in %.0fs", msg[:80], attempt, delay)
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise LLMError(f"Gemini request failed: {msg[:300]}") from exc
        raise LLMError("Gemini request failed after retries")


_REGISTRY: dict[str, Callable[..., LLMProvider]] = {"gemini": lambda s: GeminiProvider(s.gemini_api_key, s.llm_model)}


def register_provider(name: str, factory: Callable[..., LLMProvider]) -> None:
    _REGISTRY[name] = factory


def get_llm(settings) -> LLMProvider | None:
    """Return the configured provider, or None when no credential is available (retrieval-only mode)."""
    if settings.llm_provider == "gemini" and not settings.gemini_api_key:
        return None
    factory = _REGISTRY.get(settings.llm_provider)
    if factory is None:
        raise LLMError(f"Unknown LLM provider {settings.llm_provider!r}; registered: {sorted(_REGISTRY)}")
    return factory(settings)
