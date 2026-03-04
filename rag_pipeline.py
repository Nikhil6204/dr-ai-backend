"""
RAG Pipeline — ChromaDB vector search for medical knowledge.
This module handles:
  - Loading the persisted ChromaDB vector store
  - Searching relevant medical context for a given query
  - Per-user collection for their personal documents (reports, prescriptions)
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import logging

logging.basicConfig(level=logging.INFO)

# ── Embedding model (free, runs locally, no API key needed) ──────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ── Paths ─────────────────────────────────────────────────────────────────────
GLOBAL_DB_PATH = "./medical_db"          # General medical knowledge (admin PDFs)
USER_DB_PATH   = "./user_db"             # Per-user personal documents


# ── Load / create vector stores ───────────────────────────────────────────────
def get_global_vectordb():
    """Returns the global medical ChromaDB collection."""
    return Chroma(
        persist_directory=GLOBAL_DB_PATH,
        embedding_function=embedder,
        collection_name="medical_knowledge"
    )

def get_user_vectordb(user_id: str):
    """Returns a personal ChromaDB collection for a specific user."""
    return Chroma(
        persist_directory=f"{USER_DB_PATH}/{user_id}",
        embedding_function=embedder,
        collection_name=f"user_{user_id}"
    )


# ── Search ─────────────────────────────────────────────────────────────────────
def search_medical_context(query: str, k: int = 4) -> str:
    """
    Search the global medical knowledge base.
    Returns top-k relevant chunks joined as a single string.
    """
    try:
        db = get_global_vectordb()
        docs = db.similarity_search(query, k=k)

        if not docs:
            logging.info("RAG: No relevant context found.")
            return ""

        context = "\n\n".join([
            f"[Source: {doc.metadata.get('source_file', 'Unknown')}]\n{doc.page_content}"
            for doc in docs
        ])
        logging.info(f"RAG: Retrieved {len(docs)} chunks for query: {query[:60]}...")
        return context

    except Exception as e:
        logging.error(f"RAG search error: {e}")
        return ""


def search_user_context(query: str, user_id: str, k: int = 3) -> str:
    """
    Search the user's personal document collection (their uploaded reports/prescriptions).
    Returns top-k relevant chunks or empty string if no personal docs exist.
    """
    try:
        db = get_user_vectordb(user_id)
        docs = db.similarity_search(query, k=k)

        if not docs:
            return ""

        context = "\n\n".join([
            f"[Patient Document: {doc.metadata.get('source_file', 'Unknown')}]\n{doc.page_content}"
            for doc in docs
        ])
        logging.info(f"RAG: Retrieved {len(docs)} personal chunks for user {user_id}")
        return context

    except Exception as e:
        logging.warning(f"User RAG search skipped (no personal docs yet): {e}")
        return ""


def build_rag_context(query: str, user_id: str = None) -> str:
    """
    Combines global medical context + user's personal document context.
    This is the main function called by brain_of_the_doctor.py
    """
    global_ctx = search_medical_context(query)
    user_ctx   = search_user_context(query, user_id) if user_id else ""

    parts = []
    if user_ctx:
        parts.append(f"=== Patient's Own Documents ===\n{user_ctx}")
    if global_ctx:
        parts.append(f"=== Medical Knowledge Base ===\n{global_ctx}")

    return "\n\n".join(parts)


# ── Admin: Add documents to global DB ─────────────────────────────────────────
def add_chunks_to_global_db(chunks: list, doc_id: str, filename: str):
    """
    Upserts text chunks into the global ChromaDB.
    Called by the ingestion pipeline after processing a PDF.
    """
    db = get_global_vectordb()

    # Attach metadata to each chunk
    for chunk in chunks:
        chunk.metadata["doc_id"]      = doc_id
        chunk.metadata["source_file"] = filename

    db.add_documents(chunks)
    logging.info(f"RAG: Added {len(chunks)} chunks from '{filename}' (doc_id: {doc_id})")


def delete_doc_from_global_db(doc_id: str):
    """
    Removes all chunks belonging to a document from ChromaDB.
    Called when admin deletes an outdated PDF.
    """
    db = get_global_vectordb()
    db._collection.delete(where={"doc_id": doc_id})
    logging.info(f"RAG: Deleted all chunks for doc_id: {doc_id}")


# ── User: Add personal documents ──────────────────────────────────────────────
def add_user_document(chunks: list, user_id: str, filename: str):
    """
    Adds a user's personal document (report/prescription) to their private collection.
    """
    db = get_user_vectordb(user_id)

    for chunk in chunks:
        chunk.metadata["user_id"]     = user_id
        chunk.metadata["source_file"] = filename

    db.add_documents(chunks)
    logging.info(f"RAG: Added {len(chunks)} personal chunks for user {user_id} from '{filename}'")
