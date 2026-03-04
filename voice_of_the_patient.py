from dotenv import load_dotenv
load_dotenv()

import logging
import os
from groq import Groq
from langdetect import detect

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------- Speech-to-Text ----------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
stt_model = "whisper-large-v3"

#  Language map for gTTS and UI
LANGUAGE_MAP = {
    'en': 'en',
    'hi': 'hi',
    'es': 'es',
    'fr': 'fr',
    'de': 'de',
    'ar': 'ar',
    'zh-cn': 'zh-CN',
    'zh-tw': 'zh-TW',
    'ru': 'ru',
    'ja': 'ja',
    'ko': 'ko'
}

def transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY, language=None):
    """
    Transcribes an audio file using Groq's Whisper model with optional multilingual support.

    Args:
        stt_model (str): Whisper model name.
        audio_filepath (str): Path to the audio file.
        GROQ_API_KEY (str): Your Groq API key.
        language (str or None): ISO 639-1 language code (e.g., 'en', 'hi', 'es') or None for auto-detect.

    Returns:
        tuple: (transcribed_text, normalized_language)
    """
    client = Groq(api_key=GROQ_API_KEY)

    with open(audio_filepath, "rb") as audio_file:
        if language:
            #  Force transcription in a specific language
            transcription = client.audio.transcriptions.create(
                model=stt_model,
                file=audio_file,
                language=language
            )
        else:
            # Auto-detect language, no translation
            transcription = client.audio.transcriptions.create(
                model=stt_model,
                file=audio_file
            )

    transcribed_text = transcription.text

    # Detect language from text if not provided
    try:
        detected_language = detect(transcribed_text)
    except Exception:
        detected_language = "en"  # fallback

    # Normalize language code using LANGUAGE_MAP
    normalized_lang = LANGUAGE_MAP.get(detected_language.lower(), "en")

    logging.info(f" Auto-detected language: {detected_language} → Normalized: {normalized_lang}")
    return transcribed_text, normalized_lang