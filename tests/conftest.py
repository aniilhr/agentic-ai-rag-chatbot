"""Offline fakes so the pipeline can be tested without OpenAI or Pinecone keys."""

from __future__ import annotations

import dataclasses

import pytest
from langchain_core.documents import Document

from src.config import get_settings
from src.graph import GroundedAnswer, build_rag_graph

EBOOK_CHUNKS = [
    ("Agentic AI refers to AI systems that can autonomously plan, decide and act to achieve goals.", 3),
    ("Unlike rule-based automation, agents adapt to new situations and use tools dynamically.", 7),
    ("Memory lets agents retain context across steps: short-term working memory and long-term stores.", 12),
]


class FakeVectorStore:
    """Returns every chunk with a similarity driven by keyword overlap."""

    def __init__(self, off_topic_similarity: float = 0.08):
        self.off_topic_similarity = off_topic_similarity
        self.calls: list[tuple[str, int]] = []

    def similarity_search_with_score(self, query: str, k: int = 4):
        self.calls.append((query, k))
        words = set(query.lower().replace("?", "").split())
        results = []
        for text, page in EBOOK_CHUNKS:
            overlap = len(words & set(text.lower().replace(",", "").replace(".", "").split()))
            score = 0.35 + 0.1 * overlap if overlap else self.off_topic_similarity
            results.append((Document(page_content=text, metadata={"page": page, "source": "ebook.pdf"}), score))
        # deliberately unsorted: the graph must sort by similarity itself
        return results[:k]


class FakeStructuredLLM:
    def __init__(self, response: GroundedAnswer):
        self.response = response
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self.response


class FakeLLM:
    def __init__(self, response: GroundedAnswer):
        self.structured = FakeStructuredLLM(response)

    def with_structured_output(self, schema):
        assert schema is GroundedAnswer
        return self.structured


@pytest.fixture
def settings():
    return dataclasses.replace(get_settings(), top_k=3, relevance_threshold=0.25)


@pytest.fixture
def make_graph(settings):
    def _make(response: GroundedAnswer | None = None, vector_store: FakeVectorStore | None = None):
        response = response or GroundedAnswer(
            answerable=True,
            answer="Agentic AI systems autonomously plan and act [1].",
            cited_chunks=[1],
            confidence=0.9,
        )
        llm = FakeLLM(response)
        store = vector_store or FakeVectorStore()
        graph = build_rag_graph(settings=settings, vector_store=store, llm=llm)
        return graph, store, llm

    return _make
