"""
brain_of_the_doctor.py — Updated with RAG context injection.

Changes from original:
  - analyze_image_with_query now accepts rag_context parameter
  - RAG context is injected into the system prompt when available
  - Everything else (Groq, Llama 4, image handling) stays exactly the same
"""

from dotenv import load_dotenv
load_dotenv()

import os
import base64
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
model = "meta-llama/llama-4-maverick-17b-128e-instruct"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def analyze_image_with_query(
    system_prompt,
    user_input,
    encoded_image=None,
    model=model,
    language="en",
    rag_context=""          # ← NEW: injected RAG context
):
    """
    Analyze patient input (text/voice/image) with RAG-enriched context.

    Args:
        system_prompt  : Base doctor persona prompt
        user_input     : Patient's text/transcribed voice
        encoded_image  : Base64 image (optional)
        model          : Groq model name
        language       : Response language
        rag_context    : Retrieved medical knowledge from ChromaDB (optional)
    """
    client = Groq(api_key=GROQ_API_KEY)

    # ── Enrich system prompt with RAG context ─────────────────────────────────
    # If RAG found relevant context, inject it so Llama 4 answers based on
    # verified medical knowledge rather than just its training data.
    if rag_context:
        enriched_system_prompt = f"""{system_prompt}

--- Relevant Medical Knowledge (use this to inform your response) ---
{rag_context}
--- End of Medical Knowledge ---

Use the above medical knowledge where relevant. If it directly addresses the patient's concern, 
prioritize it. Always maintain your friendly doctor persona."""
    else:
        enriched_system_prompt = system_prompt

    # ── Build messages (same as before) ───────────────────────────────────────
    if encoded_image:
        messages = [
            {"role": "system", "content": enriched_system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"[Respond in {language}] {user_input}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ]
    else:
        messages = [
            {"role": "system", "content": enriched_system_prompt},
            {"role": "user", "content": [{"type": "text", "text": f"[Respond in {language}] {user_input}"}]}
        ]

    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages
    )

    return chat_completion.choices[0].message.content