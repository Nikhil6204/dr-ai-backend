"""
main.py — FastAPI backend with RAG + Admin PDF ingestion + Hinglish detection + Health News
"""

from fastapi import FastAPI, UploadFile, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS
import io, os, uuid, random, logging, json
from pathlib import Path

from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import transcribe_with_groq
from rag_pipeline import (
    build_rag_context,
    add_chunks_to_global_db,
    delete_doc_from_global_db,
    add_user_document,
    get_global_vectordb
)

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import httpx

logging.basicConfig(level=logging.INFO)

# ── NewsAPI Client ─────────────────────────────────────────────────────────────
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")

# ── Hinglish Detection ─────────────────────────────────────────────────────────
HINGLISH_WORDS = {
    "mera","meri","tera","teri","uska","uski","hum","tum","aap","mai","main",
    "hai","hain","tha","thi","the","ho","hoga","hogi","kar","karo","karna","karta",
    "karti","karte","dard","pet","sir","sar","bukhar","khana","pani","neend","sota",
    "soti","bimaar","takleef","problem","theek","nahi","nhi","bahut","bohot","thoda",
    "accha","bura","jaldi","aaj","kal","subah","raat","din","ghanta","minute",
    "dawai","dawa","doctor","hospital","sehat","body","haath","pair","aankhein",
    "naak","kaan","gala","chest","pith","jodon","haddi","chamdi","skin",
    "khujli","jalan","sujan","chot","zakhm","blood","khoon","urine","latrine",
    "ulti","vomit","dast","loose","motion","constipation","kabz","gas","acidity",
    "bp","sugar","thyroid","infection","allergy","fever","cough","cold","flu",
    "weakness","kamzori","thakaan","chakkar","behoshi","anxiety","tension","stress",
    "mere","iske","uske","humara","tumhara","apna","apni","woh","yeh","kya",
    "kaise","kyun","kab","kitna","kitni","bahot","bilkul","zaroor","abhi","phir"
}

def detect_hinglish(text: str) -> str:
    if not text:
        return None
    words = text.lower().split()
    if not words:
        return None
    matches = sum(1 for w in words if w in HINGLISH_WORDS)
    ratio   = matches / len(words)
    if ratio >= 0.25:
        logging.info(f"Hinglish detected ({matches}/{len(words)} words)")
        return "hi"
    return None


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(title="AI Doctor Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── PDF Splitter ───────────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# ── System Prompt ──────────────────────────────────────────────────────────────
system_prompt = """You are Dr. AI, a highly experienced virtual medical assistant designed to help patients understand their symptoms and health concerns. You have expertise across general medicine, internal medicine, dermatology, pediatrics, and emergency care.

IMPORTANT DISCLAIMERS YOU MUST FOLLOW:
- Always clarify you are an AI assistant, not a real licensed doctor
- Always recommend consulting a real doctor for diagnosis and treatment
- Never prescribe specific medications or exact dosages
- For emergencies (chest pain, difficulty breathing, stroke symptoms), always say "call emergency services immediately"

HOW TO RESPOND:
- Speak warmly and empathetically, like a trusted family doctor
- Always acknowledge the patient's concern before giving information
- If an image is provided, analyze it carefully and describe what you observe medically
- Do NOT say "In the image I see" — instead say "Based on what I can observe..." or "From what you've shared with me..."
- If RAG medical knowledge is provided in context, prioritize that information in your response
- Give a possible differential diagnosis when symptoms are described
- Suggest practical home remedies or lifestyle changes where appropriate
- Always end with when they should urgently seek in-person medical care

RESPONSE FORMAT:
- Respond in 3-4 sentences maximum
- Use simple, easy to understand language — avoid heavy medical jargon
- Be direct and helpful, no unnecessary filler words
- Respond in the same language the patient is speaking
- Do NOT use bullet points, numbers, or special characters in your response
- Write in one flowing paragraph

TONE:
- Compassionate and reassuring, never alarming
- Professional but approachable
- Never dismissive of symptoms, even minor ones
- If patient seems anxious, acknowledge their feelings first
"""

HEALTH_TIPS = [
    "💧 Stay hydrated — drink at least 8 glasses of water daily.",
    "😴 Sleep well — aim for 7-8 hours of quality sleep each night.",
    "🥗 Eat a balanced diet rich in vegetables, fruits, and whole grains.",
    "🚶 Stay active — even a 30-minute walk daily makes a big difference.",
    "🧘 Manage stress — try deep breathing or meditation for 5 minutes a day."
]

document_registry = {}


# ════════════════════════════════════════════════════════════════════════════════
# MAIN ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/analyze")
async def analyze(
    user_text: str = Form(...),
    image: UploadFile = None,
    language: str = Form("en"),
    user_id: str = Form(None)
):
    encoded_image = None
    rag_query     = user_text

    detected_language = language
    if language == "en":
        hinglish = detect_hinglish(user_text)
        if hinglish:
            detected_language = hinglish

    LANG_NAMES = {
        "hi":"Hindi (Devanagari script)","es":"Spanish","fr":"French",
        "de":"German","ar":"Arabic","ru":"Russian","ja":"Japanese",
        "ko":"Korean","zh-cn":"Chinese Simplified","pt":"Portuguese",
        "bn":"Bengali","mr":"Marathi","ta":"Tamil","te":"Telugu","gu":"Gujarati"
    }

    if detected_language == "hi":
        lang_instruction = (
            "\n\nCRITICAL INSTRUCTION: You MUST respond ONLY in Hindi using Devanagari script. "
            "Do NOT write a single word in English. Your entire response must be in Hindi. "
            "Example start: 'आपके लक्षणों के आधार पर...'"
        )
    elif detected_language != "en":
        lang_name = LANG_NAMES.get(detected_language, detected_language)
        lang_instruction = (
            f"\n\nCRITICAL INSTRUCTION: The user has selected {lang_name} as their language. "
            f"You MUST respond entirely in {lang_name}. Do NOT use English at all."
        )
    else:
        lang_instruction = "\n\nRespond in English."

    effective_prompt = system_prompt + lang_instruction

    if image:
        image_path = f"temp_{uuid.uuid4().hex}_{image.filename}"
        with open(image_path, "wb") as f:
            f.write(await image.read())
        encoded_image = encode_image(image_path)
        if not user_text or user_text.strip() == "Doctor, please analyze this image.":
            rag_query = "medical image analysis symptoms diagnosis"

    rag_context = build_rag_context(rag_query, user_id)

    doctor_response = analyze_image_with_query(
        effective_prompt, user_text, encoded_image,
        language=detected_language, rag_context=rag_context
    )

    if image and os.path.exists(image_path):
        os.remove(image_path)

    return {
        "doctor_response": doctor_response,
        "health_tip":      random.choice(HEALTH_TIPS),
        "language":        detected_language,
        "rag_used":        bool(rag_context)
    }


@app.post("/transcribe")
async def transcribe(audio: UploadFile):
    audio_path = f"temp_{uuid.uuid4().hex}_{audio.filename}"
    with open(audio_path, "wb") as f:
        f.write(await audio.read())
    text, lang = transcribe_with_groq(
        "whisper-large-v3", audio_path, os.environ.get("GROQ_API_KEY")
    )
    if os.path.exists(audio_path):
        os.remove(audio_path)
    return {"transcription": text, "language": lang}


@app.post("/tts")
async def tts(input_text: str = Form(...), language: str = Form("en")):
    try:
        buffer  = io.BytesIO()
        tts_obj = gTTS(text=input_text, lang=language, slow=False)
        tts_obj.write_to_fp(buffer)
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="audio/mpeg")
    except Exception as e:
        logging.warning(f"TTS failed for lang={language}, falling back to English: {e}")
        buffer  = io.BytesIO()
        tts_obj = gTTS(text=input_text, lang="en", slow=False)
        tts_obj.write_to_fp(buffer)
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="audio/mpeg")


# ════════════════════════════════════════════════════════════════════════════════
# HEALTH NEWS ENDPOINT
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/news")
async def get_health_news(request: Request):
    """Fetch real health news from NewsAPI.org with real thumbnails."""
    try:
        if not NEWS_API_KEY:
            raise HTTPException(status_code=500, detail="NEWS_API_KEY not set in environment")

        body     = await request.json()
        query    = body.get("query", "health medical")
        category = body.get("category", "health")

        # Map category keys to NewsAPI queries
        CATEGORY_QUERIES = {
            "health":    "health medical",
            "medicine":  "medicine drug treatment",
            "mental":    "mental health anxiety depression",
            "nutrition": "nutrition diet food health",
            "fitness":   "fitness exercise workout",
            "disease":   "disease virus infection outbreak",
            "cancer":    "cancer tumor oncology",
            "aimed":     "artificial intelligence medicine healthcare",
        }

        search_query = CATEGORY_QUERIES.get(category, query)

        # Call NewsAPI — everything-endpoint for broad health news
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q":            search_query,
                    "language":     "en",
                    "sortBy":       "publishedAt",   # newest first
                    "pageSize":     12,
                    "apiKey":       NEWS_API_KEY,
                },
                timeout=15.0
            )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="NewsAPI error")

        data     = response.json()
        raw      = data.get("articles", [])

        articles = []
        for a in raw:
            # Skip articles with removed content or no title
            if not a.get("title") or a["title"] == "[Removed]":
                continue
            if not a.get("url") or a["url"] == "https://removed.com":
                continue

            articles.append({
                "title":       a.get("title", ""),
                "description": a.get("description") or a.get("content", "")[:200] or "",
                "url":         a.get("url", ""),
                "image":       a.get("urlToImage") or None,
                "source":      a.get("source", {}).get("name", "News"),
                "publishedAt": a.get("publishedAt", ""),
            })

        # Already sorted newest first by NewsAPI sortBy=publishedAt
        articles = articles[:10]

        logging.info(f"NewsAPI: returned {len(articles)} articles for: {search_query}")
        return {"articles": articles}

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="NewsAPI request timed out")
    except Exception as e:
        logging.error(f"News error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════════════════════════
# MEDICAL REPORT ANALYZER ENDPOINT
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/analyze-report")
async def analyze_report(
    file_base64: str  = Form(...),
    file_type:   str  = Form(...),
    file_name:   str  = Form("report"),
    member_name: str  = Form("Patient"),
):
    """Analyze a medical report (image or PDF) and return structured findings."""
    try:
        import base64 as b64_module

        # Save temp file
        temp_path = f"temp_report_{uuid.uuid4().hex}"
        raw_bytes = b64_module.b64decode(file_base64)

        # If PDF — convert first page to image using fitz
        if "pdf" in file_type.lower():
            pdf_temp = temp_path + ".pdf"
            with open(pdf_temp, "wb") as f:
                f.write(raw_bytes)
            try:
                import fitz
                pdf_doc  = fitz.open(pdf_temp)
                page     = pdf_doc[0]
                mat      = fitz.Matrix(2.0, 2.0)
                pix      = page.get_pixmap(matrix=mat)
                img_path = temp_path + ".png"
                pix.save(img_path)
                pdf_doc.close()
                os.remove(pdf_temp)
                encoded_image = encode_image(img_path)
                os.remove(img_path)
            except Exception as pdf_err:
                logging.error(f"PDF conversion error: {pdf_err}")
                raise HTTPException(status_code=400, detail="Could not process PDF. Try uploading as an image.")
        else:
            ext       = ".jpg" if "jpeg" in file_type or "jpg" in file_type else ".png"
            img_path  = temp_path + ext
            with open(img_path, "wb") as f:
                f.write(raw_bytes)
            encoded_image = encode_image(img_path)
            os.remove(img_path)

        # Specialized medical report analysis prompt
        report_prompt = """You are a medical report analyzer AI. Analyze the uploaded medical report image carefully.

Extract ALL test results and patient details, return a structured JSON response ONLY. No markdown, no explanation outside the JSON.

Return this exact JSON structure:
{
  "report_type": "Blood Test CBC / Liver Function / Lipid Profile / Thyroid / etc",
  "report_date": "date if visible, else null",
  "overall_status": "normal OR borderline OR abnormal",
  "summary": "2-3 sentence plain English summary of the overall report for a non-medical person",
  "patient": {
    "name": "patient name from report if visible, else null",
    "age": "age if visible, else null",
    "gender": "Male or Female if visible, else null",
    "patient_id": "patient ID or registration number if visible, else null",
    "referred_by": "doctor name if visible, else null",
    "lab_name": "laboratory or hospital name if visible, else null"
  },
  "tests": [
    {
      "name": "Test name e.g. Hemoglobin",
      "value": "actual value e.g. 13.2",
      "unit": "unit e.g. g/dL",
      "normal_range": "normal range e.g. 13.5-17.5",
      "status": "normal OR high OR low OR borderline OR info",
      "explanation": "one sentence plain English explanation of what this test measures"
    }
  ],
  "recommendations": [
    "Actionable recommendation 1",
    "Actionable recommendation 2"
  ]
}

Rules:
- status must be exactly one of: normal, high, low, borderline, info
- overall_status must be exactly one of: normal, borderline, abnormal
- If a value is outside normal range → high or low
- If slightly outside → borderline
- Extract every single test visible in the report
- For patient fields: if not visible in report use null
- Keep explanations simple, avoid medical jargon
- Return ONLY the JSON object, nothing else"""

        raw_response = analyze_image_with_query(
            report_prompt,
            f"Please analyze this medical report for {member_name} and return the structured JSON.",
            encoded_image,
            language="en",
            rag_context=None
        )

        # Extract JSON from response
        text  = raw_response.strip()
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON in response")

        result = json.loads(text[start:end])
        logging.info(f"Report analyzed: {result.get('report_type')} for {member_name}")
        return result

    except json.JSONDecodeError as e:
        logging.error(f"Report JSON parse error: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse AI response")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Report analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════════════════════════
# ADMIN ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/admin/upload-pdf")
async def admin_upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = None,
    source: str = Form("Admin Upload"),
    tags: str = Form("")
):
    if not file or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a valid PDF file.")

    temp_path = f"temp_admin_{uuid.uuid4().hex}_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    doc_id = str(uuid.uuid4())
    document_registry[doc_id] = {
        "filename": file.filename, "status": "processing",
        "source": source, "tags": [t.strip() for t in tags.split(",") if t.strip()],
        "chunk_count": 0, "error": None
    }
    background_tasks.add_task(_process_pdf_background, temp_path, doc_id, file.filename, source)
    return {"message": "PDF upload received.", "doc_id": doc_id, "filename": file.filename}


def _process_pdf_background(temp_path, doc_id, filename, source):
    try:
        loader = PyMuPDFLoader(temp_path)
        docs   = loader.load()
        docs   = [d for d in docs if d.page_content.strip()]

        if not docs:
            import fitz
            pdf_doc = fitz.open(temp_path)
            docs = []
            for page in pdf_doc:
                text = page.get_text("text").strip()
                if text:
                    docs.append(Document(page_content=text, metadata={"source": filename, "page": page.number}))
            pdf_doc.close()

        if not docs:
            try:
                import fitz, pytesseract
                from PIL import Image
                pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                pdf_doc = fitz.open(temp_path)
                for page in pdf_doc:
                    mat = fitz.Matrix(2, 2)
                    pix = page.get_pixmap(matrix=mat)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    text = pytesseract.image_to_string(img).strip()
                    if text:
                        docs.append(Document(page_content=text, metadata={"source": filename, "page": page.number}))
                pdf_doc.close()
            except Exception as ocr_err:
                logging.error(f"OCR failed: {ocr_err}")

        if not docs:
            raise ValueError(f"No text extracted from '{filename}'.")

        chunks = splitter.split_documents(docs)
        chunks = [c for c in chunks if c.page_content.strip()]
        if not chunks:
            raise ValueError("No valid chunks after splitting.")

        for chunk in chunks:
            chunk.metadata["source_tag"] = source

        add_chunks_to_global_db(chunks, doc_id, filename)
        document_registry[doc_id]["status"]      = "indexed"
        document_registry[doc_id]["chunk_count"] = len(chunks)
        logging.info(f"PDF ingested: {filename} → {len(chunks)} chunks")

    except Exception as e:
        document_registry[doc_id]["status"] = "failed"
        document_registry[doc_id]["error"]  = str(e)
        logging.error(f"PDF ingestion failed: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/admin/documents")
async def list_documents():
    return {"documents": document_registry}


@app.delete("/admin/delete-pdf/{doc_id}")
async def delete_pdf(doc_id: str):
    if doc_id not in document_registry:
        raise HTTPException(status_code=404, detail="Document not found.")
    delete_doc_from_global_db(doc_id)
    filename = document_registry.pop(doc_id, {}).get("filename", doc_id)
    return {"message": f"Document '{filename}' removed from RAG."}


@app.get("/admin/rag-status")
async def rag_status():
    try:
        db    = get_global_vectordb()
        count = db._collection.count()
        return {"status": "healthy", "total_chunks": count, "total_documents": len(document_registry)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ════════════════════════════════════════════════════════════════════════════════
# USER DOCUMENT ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/user/upload-document")
async def user_upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = None,
    user_id: str = Form(...)
):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided.")
    if not file.filename.lower().endswith(".pdf"):
        return {"message": "Image files are analyzed by vision model. Only PDFs are indexed for RAG."}

    temp_path = f"temp_user_{uuid.uuid4().hex}_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    background_tasks.add_task(_process_user_pdf, temp_path, user_id, file.filename)
    return {"message": "Your document is being indexed.", "filename": file.filename}


def _process_user_pdf(temp_path, user_id, filename):
    try:
        loader = PyMuPDFLoader(temp_path)
        docs   = loader.load()
        docs   = [d for d in docs if d.page_content.strip()]

        if not docs:
            import fitz
            pdf_doc = fitz.open(temp_path)
            docs = []
            for page in pdf_doc:
                text = page.get_text("text").strip()
                if text:
                    docs.append(Document(page_content=text, metadata={"source": filename, "page": page.number}))
            pdf_doc.close()

        chunks = splitter.split_documents(docs)
        chunks = [c for c in chunks if c.page_content.strip()]
        if chunks:
            add_user_document(chunks, user_id, filename)
            logging.info(f"User doc indexed: {filename} → {len(chunks)} chunks")
        else:
            logging.warning(f"No chunks from: {filename}")
    except Exception as e:
        logging.error(f"User doc ingestion failed: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
