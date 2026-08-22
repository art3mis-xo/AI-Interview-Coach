# rag/setup.py
from pathlib import Path

from rag.loader import load_all_documents, load_job_description, split_documents
from rag.retriever import create_vector_store, create_retriever
from chains.question_generator import create_question_generator


# def setup_interview_rag(job_description_path: str):
#     """Complete setup for RAG-powered interviews from a file, folder, or default JD corpus."""

#     path = Path(job_description_path)

#     # 1. Load documents
#     print("Loading job description...")
#     if path.is_dir():
#         docs = load_all_documents(str(path))
#     else:
#         docs = load_job_description(str(path))

#     # 2. Split into chunks
#     print("Splitting into chunks...")
#     chunks = split_documents(docs, chunk_size=300, chunk_overlap=30)
#     print(f"Created {len(chunks)} chunks")

#     # 3. Create vector store
#     print("Creating embeddings and vector store...")
#     vector_store = create_vector_store(chunks)

#     # 4. Create retriever
#     retriever = create_retriever(vector_store, k=3)

#     # 5. Create question generator
#     question_generator = create_question_generator(retriever)

#     return {
#         "vector_store": vector_store,
#         "retriever": retriever,
#         "question_generator": question_generator
#     }

from rag.loader import load_all_job_descriptions
from chains.question_generator import create_question_generator
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def setup_interview_rag(directory="data/job_descriptions"):
    docs = load_all_job_descriptions(directory)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")    
    vector_store = Chroma.from_documents(splits, embedding=embeddings)
    retriever = vector_store.as_retriever()

    # Build your question generator chain using retriever
    question_generator = create_question_generator(retriever)
    return {"question_generator": question_generator, "vector_store": vector_store, "documents": splits}
