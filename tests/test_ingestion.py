from langchain_core.documents import Document

from src.ingestion import clean_text, split_documents


def test_clean_text_normalises_pdf_artifacts():
    raw = "Agen-\ntic   AI\x00 systems\n\n\n\n  act  \n autonomously"
    assert clean_text(raw) == "Agentic AI systems\n\nact\nautonomously"


def test_split_documents_chunks_with_metadata_and_stable_ids():
    body = " ".join(f"Sentence {i} about autonomous agents and planning." for i in range(200))
    pages = [
        Document(page_content=body, metadata={"page": 0, "source": "/tmp/data/Ebook-Agentic-AI.pdf"}),
        Document(page_content="   ", metadata={"page": 1, "source": "/tmp/data/Ebook-Agentic-AI.pdf"}),
    ]

    chunks = split_documents(pages, chunk_size=500, chunk_overlap=100)
    again = split_documents(pages, chunk_size=500, chunk_overlap=100)

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 500 for c in chunks)
    assert {c.metadata["page"] for c in chunks} == {1}  # blank page skipped, 1-based numbering
    assert all(c.metadata["source"] == "Ebook-Agentic-AI.pdf" for c in chunks)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(set(ids)) == len(ids)
    assert ids == [c.metadata["chunk_id"] for c in again]  # idempotent re-ingestion


def test_chunks_overlap():
    body = " ".join(f"word{i}" for i in range(400))
    chunks = split_documents([Document(page_content=body, metadata={"page": 0})], 300, 100)
    first_tail = chunks[0].page_content.split()[-3:]
    assert all(word in chunks[1].page_content for word in first_tail)
