"""Factories for the Gemini embedding and chat models, hardened against API errors.

Ingestion and retrieval both build their embeddings here, so documents and
queries are always embedded with the same model and output dimension.

Failure modes handled here:

* **Overload / rate limits** (HTTP 429 / 5xx, e.g. ``503 UNAVAILABLE``):
  every call is retried with exponential backoff, and chat calls then fall
  back to the next Gemini model, which has its own capacity.
* **Model not available for the API key** (HTTP 404): the configured chat
  models are checked against the models the key can actually use, and
  unavailable ones are skipped.
"""

from __future__ import annotations

import logging
from typing import Any

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.config import Settings

logger = logging.getLogger(__name__)

# HTTP status codes worth retrying: timeout, rate limit and server-side errors.
TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

EMBEDDING_ATTEMPTS = 5
CHAT_ATTEMPTS_PER_MODEL = 3
CHAT_TIMEOUT_SECONDS = 60.0


def _error_chain(exc: BaseException):
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def error_status_code(exc: BaseException) -> int | None:
    """HTTP status code of a Gemini API error, looking through wrapped exceptions."""
    from google.genai.errors import APIError

    for err in _error_chain(exc):
        if isinstance(err, APIError):
            return err.code
    return None


def is_transient_error(exc: BaseException) -> bool:
    """True for errors that may succeed on retry (overload, rate limit, network)."""
    import httpx

    for err in _error_chain(exc):
        if isinstance(err, (httpx.TimeoutException, httpx.NetworkError)):
            return True
    return error_status_code(exc) in TRANSIENT_STATUS_CODES


def _retrying(attempts: int):
    return retry(
        retry=retry_if_exception(is_transient_error),
        stop=stop_after_attempt(attempts),
        wait=wait_exponential_jitter(initial=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def build_embeddings(settings: Settings):
    """Gemini embeddings (RETRIEVAL_DOCUMENT for chunks, RETRIEVAL_QUERY for queries).

    The embedding model is never swapped for another one on failure: vectors
    from different models are not comparable, so transient errors are retried.
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    class RetryingGeminiEmbeddings(GoogleGenerativeAIEmbeddings):
        @_retrying(EMBEDDING_ATTEMPTS)
        def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return super().embed_documents(texts, **kwargs)

        @_retrying(EMBEDDING_ATTEMPTS)
        def embed_query(self, text: str, **kwargs: Any) -> list[float]:
            return super().embed_query(text, **kwargs)

    return RetryingGeminiEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.gemini_api_key,
        output_dimensionality=settings.embedding_dimension,
    )


# --------------------------------------------------------------------------- #
# Chat model
# --------------------------------------------------------------------------- #
def list_generation_models(api_key: str | None) -> list[str] | None:
    """Names of models this API key can call generateContent on (None if unknown)."""
    from google import genai

    try:
        client = genai.Client(api_key=api_key)
        return [
            model.name.removeprefix("models/")
            for model in client.models.list()
            if model.name and "generateContent" in (model.supported_actions or [])
        ]
    except Exception as exc:
        if error_status_code(exc) in (400, 401, 403):
            raise RuntimeError(f"GEMINI_API_KEY was rejected by the Gemini API: {exc}") from exc
        # Listing is best-effort otherwise (e.g. a network blip); runtime fallbacks still apply.
        logger.warning("Could not list Gemini models (%s); using configured models as-is", exc)
        return None


def _is_text_flash_model(name: str) -> bool:
    excluded = ("image", "tts", "audio", "live", "embedding", "thinking", "exp")
    return "flash" in name and not any(word in name for word in excluded)


def resolve_chat_models(settings: Settings, available: list[str] | None) -> list[str]:
    """Configured model + fallbacks, minus models the API key cannot use."""
    candidates = list(dict.fromkeys([settings.llm_model, *settings.llm_fallback_models]))
    if available is None:
        return candidates

    usable = [name for name in candidates if name in available]
    skipped = [name for name in candidates if name not in available]
    if skipped:
        logger.warning("Gemini model(s) not available for this API key, skipping: %s", skipped)
    if not usable:
        # None of the configured models exist for this key: use any Flash text model it has.
        usable = [name for name in available if _is_text_flash_model(name)][:3]
    if not usable:
        raise RuntimeError(
            f"None of the configured Gemini chat models {candidates} are available for this "
            "GEMINI_API_KEY. Set LLM_MODEL to a model listed at https://ai.google.dev/gemini-api/docs/models."
        )
    return usable


class FallbackChatModel:
    """Chat models tried in order; each is retried on overload before moving on."""

    def __init__(self, models: list[Any]):
        if not models:
            raise ValueError("at least one chat model is required")
        self.models = models

    @property
    def model_names(self) -> list[str]:
        return [getattr(m, "model", "?") for m in self.models]

    def with_structured_output(self, schema: Any):
        primary, *fallbacks = [m.with_structured_output(schema) for m in self.models]
        return primary.with_fallbacks(fallbacks) if fallbacks else primary


def build_llm(settings: Settings, available_models: list[str] | None = None) -> FallbackChatModel:
    """Gemini chat model(s) used by the ``generate`` node."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    if available_models is None:
        available_models = list_generation_models(settings.gemini_api_key)
    names = resolve_chat_models(settings, available_models)
    logger.info("Using Gemini chat model(s), in fallback order: %s", names)
    return FallbackChatModel(
        [
            ChatGoogleGenerativeAI(
                model=name,
                temperature=0,
                google_api_key=settings.gemini_api_key,
                max_retries=CHAT_ATTEMPTS_PER_MODEL,
                timeout=CHAT_TIMEOUT_SECONDS,
            )
            for name in names
        ]
    )
