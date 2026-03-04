import os
from gtts import gTTS
from langdetect import detect, DetectorFactory

# Ensure consistent language detection
DetectorFactory.seed = 0

# ---------------- gTTS Section ----------------
LANGUAGE_MAP = {
    'en': 'en', 'hi': 'hi', 'es': 'es', 'fr': 'fr',
    'de': 'de', 'ar': 'ar', 'zh-cn': 'zh-CN', 'zh-tw': 'zh-TW',
    'ru': 'ru', 'ja': 'ja', 'ko': 'ko'
}

def text_to_speech_with_gtts(input_text, output_filepath="output.mp3", language=None):
    try:
        # Detect language if not provided
        if language is None:
            detected_lang = detect(input_text)
            language = LANGUAGE_MAP.get(detected_lang, 'en')  # fallback to English
            print(f"🔍 Auto-detected language: {detected_lang} → Using gTTS language: {language}")

        # Generate speech and save to file
        tts = gTTS(text=input_text, lang=language, slow=False)
        tts.save(output_filepath)
        print(f"✅ Audio saved at: {os.path.abspath(output_filepath)}")

    except Exception as e:
        print(f"❌ Error in gTTS: {e}")

