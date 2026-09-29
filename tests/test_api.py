from fastapi.testclient import TestClient

import app as api
from src.config import get_settings


def client_for(graph):
    api.app.dependency_overrides[api.graph_dependency] = lambda: graph
    return TestClient(api.app)


def teardown_function():
    api.app.dependency_overrides.clear()


def test_chat_returns_answer_chunks_and_score(make_graph):
    graph, _, _ = make_graph()
    response = client_for(graph).post("/chat", json={"query": "What is Agentic AI?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Agentic AI")
    assert 0.0 < body["confidence_score"] <= 1.0
    assert body["grounded"] is True
    assert len(body["retrieved_chunks"]) == 3
    first = body["retrieved_chunks"][0]
    assert set(first) == {"rank", "text", "page", "source", "similarity_score", "cited"}
    assert first["cited"] is True


def test_chat_refuses_off_topic(make_graph):
    graph, _, _ = make_graph()
    body = client_for(graph).post("/chat", json={"query": "Who won the 2022 FIFA World Cup?"}).json()

    assert body["grounded"] is False
    assert body["confidence_score"] == 0.0
    assert "cannot answer" in body["answer"]


def test_chat_validates_input(make_graph):
    graph, _, _ = make_graph()
    client = client_for(graph)
    assert client.post("/chat", json={}).status_code == 422
    assert client.post("/chat", json={"query": ""}).status_code == 422
    assert client.post("/chat", json={"query": "   "}).status_code == 422


def test_health():
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    settings = get_settings()
    assert body["embedding_model"] == settings.embedding_model
    assert body["embedding_dimension"] == settings.embedding_dimension
    assert body["llm_model"] == settings.llm_model


def test_chat_returns_503_when_backend_unavailable(monkeypatch):
    def boom():
        raise RuntimeError("Missing required environment variable(s): GEMINI_API_KEY")

    monkeypatch.setattr(api, "get_rag_graph", boom)
    response = TestClient(api.app).post("/chat", json={"query": "What is Agentic AI?"})
    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]
