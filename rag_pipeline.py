"""
RAG Pipeline — ChromaDB vector search for medical knowledge.
Uses lazy loading to avoid blocking server startup.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import logging, os

logging.basicConfig(level=logging.INFO)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GLOBAL_DB_PATH  = "./medical_db"
USER_DB_PATH    = "./user_db"

# ── Lazy embedder — loads ONLY on first request, not on startup ───────────────
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        logging.info("RAG: Loading embedding model (first request)...")
        _embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        logging.info("RAG: Embedding model loaded!")
    return _embedder


# ── Vector stores ─────────────────────────────────────────────────────────────
def get_global_vectordb():
    return Chroma(
        persist_directory=GLOBAL_DB_PATH,
        embedding_function=get_embedder(),
        collection_name="medical_knowledge"
    )

def get_user_vectordb(user_id: str):
    return Chroma(
        persist_directory=f"{USER_DB_PATH}/{user_id}",
        embedding_function=get_embedder(),
        collection_name=f"user_{user_id}"
    )


# ── Search ────────────────────────────────────────────────────────────────────
def search_medical_context(query: str, k: int = 4) -> str:
    try:
        db   = get_global_vectordb()
        docs = db.similarity_search(query, k=k)
        if not docs:
            return ""
        return "\n\n".join([
            f"[Source: {doc.metadata.get('source_file','Unknown')}]\n{doc.page_content}"
            for doc in docs
        ])
    except Exception as e:
        logging.error(f"RAG search error: {e}")
        return ""


def search_user_context(query: str, user_id: str, k: int = 3) -> str:
    try:
        db   = get_user_vectordb(user_id)
        docs = db.similarity_search(query, k=k)
        if not docs:
            return ""
        return "\n\n".join([
            f"[Patient Document: {doc.metadata.get('source_file','Unknown')}]\n{doc.page_content}"
            for doc in docs
        ])
    except Exception as e:
        logging.warning(f"User RAG skipped: {e}")
        return ""


def build_rag_context(query: str, user_id: str = None) -> str:
    try:
        global_ctx = search_medical_context(query)
        user_ctx   = search_user_context(query, user_id) if user_id else ""
        parts = []
        if user_ctx:   parts.append(f"=== Patient's Own Documents ===\n{user_ctx}")
        if global_ctx: parts.append(f"=== Medical Knowledge Base ===\n{global_ctx}")
        return "\n\n".join(parts)
    except Exception as e:
        logging.error(f"RAG context error: {e}")
        return ""


# ── Admin: Add/Delete documents ───────────────────────────────────────────────
def add_chunks_to_global_db(chunks: list, doc_id: str, filename: str):
    db = get_global_vectordb()
    for chunk in chunks:
        chunk.metadata["doc_id"]      = doc_id
        chunk.metadata["source_file"] = filename
    db.add_documents(chunks)
    logging.info(f"RAG: Added {len(chunks)} chunks from '{filename}'")


def delete_doc_from_global_db(doc_id: str):
    db = get_global_vectordb()
    db._collection.delete(where={"doc_id": doc_id})
    logging.info(f"RAG: Deleted chunks for doc_id: {doc_id}")


# ── User: Personal documents ──────────────────────────────────────────────────
def add_user_document(chunks: list, user_id: str, filename: str):
    db = get_user_vectordb(user_id)
    for chunk in chunks:
        chunk.metadata["user_id"]     = user_id
        chunk.metadata["source_file"] = filename
    db.add_documents(chunks)
    logging.info(f"RAG: Added {len(chunks)} personal chunks for user {user_id}")