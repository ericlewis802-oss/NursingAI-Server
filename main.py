from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import requests
import os
import json

app = FastAPI()

POE_ACCESS_KEY = os.environ.get("POE_ACCESS_KEY")
POE_API_URL = "https://api.poe.com/bot/"


def call_poe_model(model, messages, attachments=None, search=False):
    """Send a request to a Poe model and return output text."""
    url = POE_API_URL + model

    payload = {
        "messages": messages,
        "search": search
    }

    if attachments:
        payload["attachments"] = attachments

    headers = {
        "Authorization": f"Bearer {POE_ACCESS_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    return data.get("output_text", "")


# ---------------------------------------------------------
# HEALTH CHECK ENDPOINT (Poe requires GET / to succeed)
# ---------------------------------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "NursingAI server running"}


# ---------------------------------------------------------
# MAIN WEBHOOK ENDPOINT
# ---------------------------------------------------------
@app.post("/")
async def poe_webhook(request: Request):
    body = await request.json()

    # Ignore empty POST requests (health checks)
    if not body or body == {}:
        return {"response": "OK"}

    # Ignore requests with no message and no attachments
    if not body.get("message") and not body.get("attachments"):
        return {"response": "OK"}

    user_message = body.get("message", "")
    attachments = body.get("attachments", [])
    user_context = body.get("user_context", "")

    # -----------------------------------------------------
    # NO ATTACHMENTS → CONVERSATION MODE
    # -----------------------------------------------------
    if not attachments:
        messages = [{"role": "user", "content": user_message}]
        reply = call_poe_model("gemini-2.5-flash", messages)
        return {"response": reply}

    # -----------------------------------------------------
    # STEP 1 — OCR EXTRACTION USING GPT‑4o
    # -----------------------------------------------------
    extracted_text = call_poe_model(
        "gpt-4o",
        [{"role": "system", "content": "Extract all text from the attachments. Do NOT summarize."}],
        attachments=attachments
    )

    # -----------------------------------------------------
    # STEP 2 — ANALYSIS USING GPT‑5.1
    # -----------------------------------------------------
    analysis_prompt = f"""
Analyze this medical content and return ONLY JSON as described.

Content:
{extracted_text}

User context:
{user_context}

Return ONLY clean JSON with no explanations outside the JSON object.
"""

    try:
        analysis_json_raw = call_poe_model(
            "gpt-5.1",
            [{"role": "user", "content": analysis_prompt}]
        )
        analysis = json.loads(analysis_json_raw)
        needs_web = analysis.get("needs_web_search", True)
    except:
        analysis_json_raw = "{}"
        needs_web = True

    # -----------------------------------------------------
    # STEP 3 — FLASHCARD GENERATION USING GPT‑5.1
    # -----------------------------------------------------
    flashcard_prompt = f"""
SYSTEM PROMPT:
You are NursingAI — generate STRICTLY formatted medical flashcards.
Follow ALL flashcard rules the user previously defined.

<analysis_json>
{analysis_json_raw}
</analysis_json>

<content>
{extracted_text}
</content>

BEGIN FLASHCARD GENERATION NOW.
"""

    flashcards = call_poe_model(
        "gpt-5.1",
        [{"role": "user", "content": flashcard_prompt}],
        search=needs_web
    )

    return {"response": flashcards}
