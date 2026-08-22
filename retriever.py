# rag/retriever.py
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

def create_vector_store(documents: List[Document], persist_directory: str = None):
    """Create a vector store from documents."""

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")   

    if persist_directory:
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
    else:
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings
        )

    return vector_store

def load_vector_store(persist_directory: str):
    """Load an existing vector store."""

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")   

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

def create_retriever(vector_store, k: int = 4):
    """Create a retriever that finds similar documents."""

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )