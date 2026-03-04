"""
ingest_pdfs.py — One-time (and repeated) PDF ingestion script.

Usage:
  # Ingest all PDFs from a folder
  python ingest_pdfs.py --folder ./medical_pdfs

  # Ingest a single PDF
  python ingest_pdfs.py --file ./medical_pdfs/cardiology.pdf

Free Medical PDF Sources:
  - PMC Open Access: https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/
  - MedlinePlus:     https://medlineplus.gov
  - CDC:             https://www.cdc.gov/healthtopics.html
  - NICE Guidelines: https://www.nice.org.uk/guidance
"""

import os
import uuid
import argparse
import logging
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from rag_pipeline import add_chunks_to_global_db

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ── Text splitter ──────────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)


def extract_text_with_ocr(pdf_path: Path) -> list:
    """
    Fallback OCR extractor for image-based PDFs (e.g. saved via Ctrl+P from browser).
    Uses Tesseract OCR to extract text from each page as an image.
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract

        # Windows Tesseract path — update if installed elsewhere
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        logging.info(f"🔍 Running OCR on: {pdf_path.name}")
        images = convert_from_path(str(pdf_path))

        full_text = ""
        for i, img in enumerate(images):
            page_text = pytesseract.image_to_string(img)
            full_text += page_text + "\n"
            logging.info(f"   OCR page {i+1}/{len(images)} done")

        if full_text.strip():
            return [Document(
                page_content=full_text,
                metadata={"source": str(pdf_path), "ocr": True}
            )]
        else:
            logging.warning(f"OCR found no text in: {pdf_path.name}")
            return []

    except ImportError:
        logging.error("OCR packages missing. Run: pip install pytesseract pdf2image pillow")
        logging.error("Also install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
        return []
    except Exception as e:
        logging.error(f"OCR failed for {pdf_path.name}: {e}")
        return []


def ingest_single_pdf(pdf_path: str, source_tag: str = "Manual Upload"):
    """
    Processes a single PDF and adds it to ChromaDB.
    Automatically falls back to OCR if PDF is image-based.
    Returns (doc_id, chunk_count).
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        logging.error(f"File not found: {pdf_path}")
        return None, 0

    if pdf_path.suffix.lower() != ".pdf":
        logging.warning(f"Skipping non-PDF file: {pdf_path.name}")
        return None, 0

    logging.info(f"Processing: {pdf_path.name}")

    try:
        # ── Step 1: Try normal text extraction ────────────────────────────────
        loader = PyMuPDFLoader(str(pdf_path))
        docs   = loader.load()

        # ── Step 2: Check if text was actually extracted ───────────────────────
        # Image-based PDFs return docs with empty or whitespace-only content
        has_text = docs and any(len(doc.page_content.strip()) > 20 for doc in docs)

        if not has_text:
            logging.info(f"⚠️ No text found in '{pdf_path.name}' — trying OCR fallback...")
            docs = extract_text_with_ocr(pdf_path)

        if not docs:
            logging.warning(f"❌ Could not extract any text from: {pdf_path.name}")
            return None, 0

        # ── Step 3: Split into chunks ──────────────────────────────────────────
        chunks = splitter.split_documents(docs)

        if not chunks:
            logging.warning(f"No chunks generated for: {pdf_path.name}")
            return None, 0

        # ── Step 4: Add metadata ───────────────────────────────────────────────
        for chunk in chunks:
            chunk.metadata["source_tag"] = source_tag

        # ── Step 5: Store in ChromaDB ──────────────────────────────────────────
        doc_id = str(uuid.uuid4())
        add_chunks_to_global_db(chunks, doc_id, pdf_path.name)

        logging.info(f"✅ Ingested '{pdf_path.name}' → {len(chunks)} chunks (doc_id: {doc_id})")
        return doc_id, len(chunks)

    except Exception as e:
        logging.error(f"❌ Failed to ingest '{pdf_path.name}': {e}")
        return None, 0


def ingest_folder(folder_path: str, source_tag: str = "Bulk Upload"):
    """Ingests all PDFs in a folder recursively."""
    folder = Path(folder_path)

    if not folder.exists():
        logging.error(f"Folder not found: {folder_path}")
        return

    pdf_files = list(folder.rglob("*.pdf"))

    if not pdf_files:
        logging.warning(f"No PDFs found in: {folder_path}")
        return

    logging.info(f"Found {len(pdf_files)} PDFs in '{folder_path}'. Starting ingestion...")

    success_count = 0
    fail_count    = 0
    total_chunks  = 0

    for pdf_file in pdf_files:
        doc_id, chunk_count = ingest_single_pdf(str(pdf_file), source_tag)
        if doc_id:
            success_count += 1
            total_chunks  += chunk_count
        else:
            fail_count += 1

    logging.info(f"""
    ─────────────────────────────
    Ingestion Complete!
    ✅ Success : {success_count} PDFs
    ❌ Failed  : {fail_count} PDFs
    📦 Total chunks added: {total_chunks}
    ─────────────────────────────
    """)


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest medical PDFs into ChromaDB")
    parser.add_argument("--file",   type=str, help="Path to a single PDF file")
    parser.add_argument("--folder", type=str, help="Path to folder containing PDFs")
    parser.add_argument("--source", type=str, default="Manual Upload", help="Source tag e.g. 'WHO Guidelines'")
    args = parser.parse_args()

    if args.file:
        ingest_single_pdf(args.file, args.source)
    elif args.folder:
        ingest_folder(args.folder, args.source)
    else:
        print("Usage:")
        print("  python ingest_pdfs.py --file ./medical_pdfs/report.pdf")
        print("  python ingest_pdfs.py --folder ./medical_pdfs --source 'CDC Guidelines'")
