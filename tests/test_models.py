"""Resilience of the Gemini layer: retries, model fallbacks and error classification."""

import dataclasses
import time

import pytest
from google.genai.errors import ClientError, ServerError
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError

from src.models import (
    FallbackChatModel,
    build_embeddings,
    build_llm,
    is_transient_error,
    list_generation_models,
    resolve_chat_models,
)


def overloaded() -> ServerError:
    return ServerError(503, {"error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}})


def not_found() -> ClientError:
    return ClientError(404, {"error": {"code": 404, "message": "models/x is not found", "status": "NOT_FOUND"}})


def wrapped(cause: Exception) -> Exception:
    try:
        raise GoogleGenerativeAIError(f"Error embedding content: {cause}") from cause
    except GoogleGenerativeAIError as exc:
        return exc


def test_transient_error_classification():
    assert is_transient_error(overloaded())
    assert is_transient_error(wrapped(overloaded()))  # as raised by the embeddings wrapper
    assert is_transient_error(ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}}))
    assert not is_transient_error(not_found())
    assert not is_transient_error(ValueError("bad input"))


def test_resolve_chat_models_skips_models_missing_for_the_key(settings):
    settings = dataclasses.replace(
        settings, llm_model="gemini-2.5-flash", llm_fallback_models=("gemini-flash-latest", "gemini-2.5-flash-lite")
    )
    available = ["gemini-flash-latest", "gemini-2.5-flash-lite", "gemini-2.5-pro"]
    assert resolve_chat_models(settings, available) == ["gemini-flash-latest", "gemini-2.5-flash-lite"]
    # model listing failed: keep the configured order and rely on runtime fallbacks
    assert resolve_chat_models(settings, None)[0] == "gemini-2.5-flash"


def test_resolve_chat_models_discovers_a_flash_model_when_none_configured_exist(settings):
    settings = dataclasses.replace(settings, llm_model="gemini-2.5-flash", llm_fallback_models=())
    available = ["gemini-2.5-pro", "gemini-3-flash", "gemini-3-flash-image", "gemini-embedding-001"]
    assert resolve_chat_models(settings, available) == ["gemini-3-flash"]

    with pytest.raises(RuntimeError, match="LLM_MODEL"):
        resolve_chat_models(settings, ["gemini-embedding-001"])


class _FakeChat:
    def __init__(self, model, error=None):
        self.model = model
        self.error = error
        self.calls = 0

    def with_structured_output(self, schema):
        def run(_):
            self.calls += 1
            if self.error:
                raise self.error
            return f"answer from {self.model}"

        return RunnableLambda(run)


def test_overloaded_model_falls_back_to_next_model():
    busy = _FakeChat("gemini-2.5-flash", overloaded())
    missing = _FakeChat("gemini-flash-latest", not_found())
    healthy = _FakeChat("gemini-2.5-flash-lite")
    structured = FallbackChatModel([busy, missing, healthy]).with_structured_output(object)

    assert structured.invoke("q") == "answer from gemini-2.5-flash-lite"
    assert (busy.calls, missing.calls, healthy.calls) == (1, 1, 1)


def test_build_llm_uses_only_available_models(settings):
    settings = dataclasses.replace(
        settings, gemini_api_key="test", llm_model="gemini-2.5-flash", llm_fallback_models=("gemini-flash-latest",)
    )
    llm = build_llm(settings, available_models=["gemini-flash-latest"])
    assert llm.model_names == ["gemini-flash-latest"]


def test_embeddings_retry_transient_errors(settings, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    outcomes = [wrapped(overloaded()), wrapped(overloaded()), [0.1, 0.2]]

    def flaky(self, text, **kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(GoogleGenerativeAIEmbeddings, "embed_query", flaky)
    embeddings = build_embeddings(dataclasses.replace(settings, gemini_api_key="test"))
    assert embeddings.embed_query("q") == [0.1, 0.2]
    assert outcomes == []


def test_embeddings_do_not_retry_permanent_errors(settings, monkeypatch):
    calls = []

    def missing(self, text, **kwargs):
        calls.append(text)
        raise wrapped(not_found())

    monkeypatch.setattr(GoogleGenerativeAIEmbeddings, "embed_query", missing)
    embeddings = build_embeddings(dataclasses.replace(settings, gemini_api_key="test"))
    with pytest.raises(GoogleGenerativeAIError):
        embeddings.embed_query("q")
    assert len(calls) == 1


def test_rejected_api_key_fails_fast(monkeypatch):
    from google import genai

    class _Client:
        def __init__(self, api_key=None):
            self.models = self

        def list(self):
            raise ClientError(400, {"error": {"code": 400, "message": "API key not valid.", "status": "INVALID_ARGUMENT"}})

    monkeypatch.setattr(genai, "Client", _Client)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY was rejected"):
        list_generation_models("bad-key")
