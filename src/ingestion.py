"""ETL pipeline: PDF -> cleaned pages -> overlapping chunks -> Pinecone.

Run as a module::

    python -m src.ingestion              # download (if missing) + ingest
    python -m src.ingestion --reset      # wipe the namespace first
    python -m src.ingestion --dry-run    # load + chunk only, no API calls

Chunk ids are derived from a hash of (page, position, text), so re-running
ingestion overwrites the same vectors instead of creating duplicates.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import PDF_DRIVE_URL, Settings, get_settings

logger = logging.getLogger(__name__)

UPSERT_BATCH_SIZE = 100


# --------------------------------------------------------------------------- #
# Extract
# --------------------------------------------------------------------------- #
def download_pdf(destination: Path, url: str = PDF_DRIVE_URL) -> Path:
    """Download the eBook from Google Drive unless it already exists locally."""
    if destination.exists() and destination.stat().st_size > 0:
        logger.info("PDF already present at %s", destination)
        return destination

    import gdown  # imported lazily: only needed for the first run

    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading eBook from Google Drive to %s", destination)
    result = gdown.download(url, str(destination), quiet=False)
    if not result or not destination.exists():
        raise RuntimeError(
            "Could not download the PDF automatically. Download it manually from "
            "https://drive.google.com/file/d/15VLphKcY23_fpYxN62UEQRri_psRVfP9/view "
            f"and save it as {destination}"
        )
    return destination


def load_pdf(pdf_path: Path) -> list[Document]:
    """Load one Document per PDF page using LangChain's PyPDFLoader."""
    from langchain_community.document_loaders import PyPDFLoader

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")
    pages = PyPDFLoader(str(pdf_path)).load()
    logger.info("Loaded %d pages from %s", len(pages), pdf_path.name)
    return pages


# --------------------------------------------------------------------------- #
# Transform
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Normalise PDF extraction artefacts without changing the wording."""
    text = text.replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_documents(
    pages: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[Document]:
    """Clean every page and split it into overlapping chunks with rich metadata."""
    cleaned = []
    for page in pages:
        text = clean_text(page.page_content)
        if not text:
            continue  # skip blank / image-only pages
        page_index = int(page.metadata.get("page", 0))
        cleaned.append(
            Document(
                page_content=text,
                metadata={
                    "source": Path(str(page.metadata.get("source", "ebook.pdf"))).name,
                    "page": page_index + 1,  # human-friendly 1-based page number
                },
            )
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(cleaned)
    for chunk in chunks:
        chunk.metadata["chunk_id"] = make_chunk_id(chunk)
    logger.info("Split %d pages into %d chunks", len(cleaned), len(chunks))
    return chunks


def make_chunk_id(chunk: Document) -> str:
    key = f"{chunk.metadata.get('page')}:{chunk.metadata.get('start_index')}:{chunk.page_content}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def ensure_index(settings: Settings):
    """Create the serverless Pinecone index (1536-d, cosine) if it doesn't exist."""
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing = set(pc.list_indexes().names())
    if settings.pinecone_index_name not in existing:
        logger.info(
            "Creating Pinecone index '%s' (dim=%d, metric=cosine)",
            settings.pinecone_index_name,
            settings.embedding_dimension,
        )
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
        while not pc.describe_index(settings.pinecone_index_name).status["ready"]:
            time.sleep(1)
    else:
        description = pc.describe_index(settings.pinecone_index_name)
        if description.dimension != settings.embedding_dimension:
            raise RuntimeError(
                f"Index '{settings.pinecone_index_name}' has dimension {description.dimension}, "
                f"but {settings.embedding_model} produces {settings.embedding_dimension}. "
                "Use a different PINECONE_INDEX_NAME or delete the index."
            )
    return pc.Index(settings.pinecone_index_name)


def upsert_chunks(chunks: list[Document], settings: Settings, reset: bool = False) -> int:
    """Embed chunks with OpenAI and upsert them into Pinecone in batches."""
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore

    index = ensure_index(settings)
    if reset:
        try:
            index.delete(delete_all=True, namespace=settings.pinecone_namespace)
            logger.info("Cleared namespace '%s'", settings.pinecone_namespace)
        except Exception:  # namespace does not exist yet
            logger.info("Namespace '%s' is empty; nothing to reset", settings.pinecone_namespace)

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )
    vector_store = PineconeVectorStore(
        index=index, embedding=embeddings, namespace=settings.pinecone_namespace
    )

    for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
        batch = chunks[start : start + UPSERT_BATCH_SIZE]
        vector_store.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])
        logger.info("Upserted %d/%d chunks", start + len(batch), len(chunks))
    return len(chunks)


def run_ingestion(settings: Settings | None = None, reset: bool = False, dry_run: bool = False) -> int:
    settings = settings or get_settings()
    pdf_path = download_pdf(settings.pdf_path)
    pages = load_pdf(pdf_path)
    chunks = split_documents(pages, settings.chunk_size, settings.chunk_overlap)

    if dry_run:
        lengths = [len(c.page_content) for c in chunks]
        logger.info(
            "Dry run: %d chunks (avg %.0f chars, max %d). Nothing was uploaded.",
            len(chunks),
            sum(lengths) / max(len(lengths), 1),
            max(lengths, default=0),
        )
        return len(chunks)

    settings.require_api_keys()
    count = upsert_chunks(chunks, settings, reset=reset)
    logger.info(
        "Ingestion complete: %d chunks in index '%s' (namespace '%s')",
        count,
        settings.pinecone_index_name,
        settings.pinecone_namespace,
    )
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the Agentic AI eBook into Pinecone.")
    parser.add_argument("--reset", action="store_true", help="delete existing vectors in the namespace first")
    parser.add_argument("--dry-run", action="store_true", help="load and chunk the PDF without calling any API")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_ingestion(reset=args.reset, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
