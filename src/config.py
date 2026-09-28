"""Environment setup and project-wide constants.

All tunables are read from environment variables (optionally loaded from a
``.env`` file at the project root) so the same code runs locally, in CI and
in a container without modification.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Google Drive file id of the Agentic AI eBook provided with the assignment.
PDF_DRIVE_FILE_ID = "15VLphKcY23_fpYxN62UEQRri_psRVfP9"
PDF_DRIVE_URL = f"https://drive.google.com/uc?id={PDF_DRIVE_FILE_ID}"

# Returned verbatim whenever the document cannot support an answer.
REFUSAL_MESSAGE = (
    "I cannot answer this based on the provided document. "
    "The Agentic AI eBook does not contain information about this topic."
)


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    pinecone_api_key: str | None
    pinecone_index_name: str
    pinecone_namespace: str
    pinecone_cloud: str
    pinecone_region: str
    embedding_model: str
    embedding_dimension: int
    llm_model: str
    pdf_path: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    relevance_threshold: float
    # Cosine-similarity range mapped linearly onto a 0..1 retrieval confidence.
    # text-embedding-3-small rarely exceeds ~0.7 even for near-paraphrases, so
    # raw cosine values would make every answer look uncertain.
    similarity_floor: float
    similarity_ceiling: float

    def require_api_keys(self) -> None:
        missing = [
            name
            for name, value in (
                ("OPENAI_API_KEY", self.openai_api_key),
                ("PINECONE_API_KEY", self.pinecone_api_key),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"Missing required environment variable(s): {', '.join(missing)}. "
                "Copy .env.example to .env and fill in your keys."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    pdf_path = Path(_env("PDF_PATH", "data/Ebook-Agentic-AI.pdf"))
    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_index_name=_env("PINECONE_INDEX_NAME", "agentic-ai-index"),
        pinecone_namespace=_env("PINECONE_NAMESPACE", "agentic-ai-ebook"),
        pinecone_cloud=_env("PINECONE_CLOUD", "aws"),
        pinecone_region=_env("PINECONE_REGION", "us-east-1"),
        embedding_model=_env("EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_dimension=int(_env("EMBEDDING_DIMENSION", "1536")),
        llm_model=_env("LLM_MODEL", "gpt-4o-mini"),
        pdf_path=pdf_path,
        chunk_size=int(_env("CHUNK_SIZE", "1000")),
        chunk_overlap=int(_env("CHUNK_OVERLAP", "200")),
        top_k=int(_env("TOP_K", "4")),
        relevance_threshold=float(_env("RELEVANCE_THRESHOLD", "0.25")),
        similarity_floor=float(_env("SIMILARITY_FLOOR", "0.20")),
        similarity_ceiling=float(_env("SIMILARITY_CEILING", "0.65")),
    )
