# rag/loader.py
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover
    from langchain.text_splitter import RecursiveCharacterTextSplitter


def create_docs_from_text(text: str, metadata: dict | None = None):
    """Create one Document from raw JD text entered in the UI."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    return [Document(page_content=cleaned, metadata=metadata or {"source": "ui_job_description", "type": "job_description"})]


def load_job_description(file_path: str):
    """Load a single job description file."""

    path = Path(file_path)

    if path.suffix == '.pdf':
        loader = PyPDFLoader(file_path)
    elif path.suffix == '.docx':
        loader = Docx2txtLoader(file_path)
    else:  # Default to text
        loader = TextLoader(file_path)

    documents = loader.load()

    # Add metadata
    for doc in documents:
        doc.metadata['source'] = path.name
        doc.metadata['type'] = 'job_description'

    return documents


def load_all_documents(directory: str):
    """Load all documents from a directory."""

    loader = DirectoryLoader(
        directory,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    return loader.load()


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into smaller chunks for embedding."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    return splitter.split_documents(documents)

import glob
from langchain_community.document_loaders import TextLoader

def load_all_job_descriptions(directory="data/job_descriptions"):
    docs = []
    for file in glob.glob(f"{directory}/*.txt"):
        loader = TextLoader(file)
        docs.extend(loader.load())
    return docs
