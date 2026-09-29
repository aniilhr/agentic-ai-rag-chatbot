import dataclasses
from types import SimpleNamespace

import pytest

from src.config import REFUSAL_MESSAGE
from src.graph import GroundedAnswer, connect_vector_store, format_context, normalise_similarity


def test_grounded_answer_flows_through_generate(make_graph):
    graph, store, llm = make_graph()
    result = graph.invoke({"question": "What is Agentic AI?"})

    assert result["grounded"] is True
    assert result["answer"].startswith("Agentic AI systems")
    assert result["cited_chunks"] == [1]
    assert 0.0 < result["score"] <= 1.0
    assert len(result["context"]) == 3
    assert store.calls == [("What is Agentic AI?", 3)]
    assert len(llm.structured.calls) == 1


def test_context_is_sorted_by_similarity_and_ranked(make_graph):
    graph, _, _ = make_graph()
    result = graph.invoke({"question": "What role does memory play for agents?"})

    sims = [c["similarity"] for c in result["context"]]
    assert sims == sorted(sims, reverse=True)
    assert [c["rank"] for c in result["context"]] == [1, 2, 3]
    assert result["context"][0]["page"] == 12


def test_prompt_contains_numbered_context_and_question(make_graph):
    graph, _, llm = make_graph()
    graph.invoke({"question": "What is Agentic AI?"})

    system, human = llm.structured.calls[0]
    assert "ONLY" in system.content
    assert "<context>" in human.content and "[1] (page" in human.content
    assert human.content.rstrip().endswith("Question: What is Agentic AI?")


def test_off_topic_question_is_refused_without_calling_llm(make_graph):
    graph, _, llm = make_graph()
    result = graph.invoke({"question": "Who won the 2022 FIFA World Cup?"})

    assert result["grounded"] is False
    assert result["answer"] == REFUSAL_MESSAGE
    assert result["score"] == 0.0
    assert llm.structured.calls == []  # relevance gate short-circuited the LLM
    assert len(result["context"]) == 3  # retrieved chunks are still returned for transparency


def test_llm_declining_produces_refusal(make_graph):
    graph, _, llm = make_graph(
        GroundedAnswer(answerable=False, answer="", cited_chunks=[], confidence=0.1)
    )
    result = graph.invoke({"question": "What is Agentic AI pricing?"})

    assert len(llm.structured.calls) == 1
    assert result["grounded"] is False
    assert result["answer"] == REFUSAL_MESSAGE
    assert result["score"] == 0.0


def test_invalid_citations_are_dropped_and_penalised(make_graph):
    cited_graph, _, _ = make_graph()
    uncited_graph, _, _ = make_graph(
        GroundedAnswer(answerable=True, answer="Agents act.", cited_chunks=[42], confidence=0.9)
    )
    question = "What is Agentic AI?"
    cited = cited_graph.invoke({"question": question})
    uncited = uncited_graph.invoke({"question": question})

    assert uncited["cited_chunks"] == []
    assert uncited["score"] < cited["score"]


def test_normalise_similarity_clamps():
    assert normalise_similarity(0.1, 0.2, 0.6) == 0.0
    assert normalise_similarity(0.9, 0.2, 0.6) == 1.0
    assert abs(normalise_similarity(0.4, 0.2, 0.6) - 0.5) < 1e-9


def test_format_context_handles_missing_page():
    text = format_context([{"rank": 1, "text": "x", "page": None, "source": "", "similarity": 0.5}])
    assert text == "[1] (page ?)\nx"


class _FakeIndexList:
    def __init__(self, names):
        self._names = names

    def names(self):
        return self._names


class _FakePinecone:
    """``indexes`` maps index name -> (dimension, vectors in the agentic-ai-ebook namespace)."""

    def __init__(self, indexes):
        self._indexes = indexes

    def __call__(self, api_key=None):
        return self

    def list_indexes(self):
        return _FakeIndexList(list(self._indexes))

    def describe_index(self, name):
        return SimpleNamespace(dimension=self._indexes[name][0])

    def Index(self, name):
        stats = SimpleNamespace(
            namespaces={"agentic-ai-ebook": SimpleNamespace(vector_count=self._indexes[name][1])}
        )
        return SimpleNamespace(name=name, describe_index_stats=lambda: stats)


OPENAI_INDEX = ("agentic-ai-index", (1536, 120))
GEMINI_INDEX = ("agentic-ai-gemini-index", (3072, 120))


@pytest.fixture
def connect(settings, monkeypatch):
    import langchain_pinecone  # import before patching so it sees the real Pinecone class
    import pinecone

    monkeypatch.setattr(
        langchain_pinecone, "PineconeVectorStore", lambda index, embedding, namespace: SimpleNamespace(index=index)
    )

    def _connect(indexes, index_name="agentic-ai-gemini"):
        monkeypatch.setattr(pinecone, "Pinecone", _FakePinecone(dict(indexes)))
        return connect_vector_store(
            dataclasses.replace(
                settings, pinecone_index_name=index_name, embedding_dimension=3072,
                pinecone_namespace="agentic-ai-ebook", gemini_api_key="test",
            )
        )

    return _connect


@pytest.mark.parametrize(
    "configured", ["agentic-ai-gemini", "agentic-ai-index"], ids=["missing", "wrong-dimension"]
)
def test_falls_back_to_the_only_usable_gemini_index(connect, configured, caplog):
    store = connect([OPENAI_INDEX, GEMINI_INDEX], index_name=configured)
    assert store.index.name == "agentic-ai-gemini-index"
    assert "PINECONE_INDEX_NAME=agentic-ai-gemini-index" in caplog.text


def test_uses_configured_index_when_usable(connect):
    other = ("agentic-ai-gemini-v2", (3072, 50))
    store = connect([GEMINI_INDEX, other], index_name="agentic-ai-gemini-index")
    assert store.index.name == "agentic-ai-gemini-index"


@pytest.mark.parametrize(
    "indexes, configured, message",
    [
        ([OPENAI_INDEX], "agentic-ai-gemini", "does not exist.*src.ingestion"),
        ([OPENAI_INDEX], "agentic-ai-index", "dimension 1536.*src.ingestion"),
        ([("agentic-ai-gemini-index", (3072, 0))], "agentic-ai-gemini-index", "no vectors.*src.ingestion"),
        (
            [GEMINI_INDEX, ("agentic-ai-gemini-v2", (3072, 50))],
            "agentic-ai-gemini",
            r"Set PINECONE_INDEX_NAME to one of \['agentic-ai-gemini-index', 'agentic-ai-gemini-v2'\]",
        ),
    ],
    ids=["missing", "wrong-dimension", "empty", "ambiguous"],
)
def test_explains_when_no_usable_index(connect, indexes, configured, message):
    with pytest.raises(RuntimeError, match=message):
        connect(indexes, index_name=configured)
