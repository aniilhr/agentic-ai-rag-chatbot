"""Factories for the Gemini embedding and chat models.

Ingestion and retrieval both build their embeddings here, so documents and
queries are always embedded with the same model and output dimension.
"""

from __future__ import annotations

from src.config import Settings


def build_embeddings(settings: Settings):
    """Gemini embeddings (RETRIEVAL_DOCUMENT for chunks, RETRIEVAL_QUERY for queries)."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.gemini_api_key,
        output_dimensionality=settings.embedding_dimension,
    )


def build_llm(settings: Settings):
    """Gemini chat model used by the ``generate`` node."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.llm_model, temperature=0, google_api_key=settings.gemini_api_key
    )
