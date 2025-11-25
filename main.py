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

    response = requests.post(
        POE_API_URL + model,
        headers=headers,
        json=payload
    )

    response.raise_for_status()
    data = response.json()
    return data.get("output_text", "")


@app.post("/")
async def poe_webhook(request: Request):
    """Main webhook called by Poe."""
    body = await request.json()

    user_message = body.get("message", "")
    attachments = body.get("attachments", [])
    user_context = body.get("user_context", "")

    # No attachments → conversation mode
    if not attachments:
        messages = [{"role": "user", "content": user_message}]
        reply = call_poe_model("gemini-2.5-flash", messages)
        return {"response": reply}

    # Step 1: OCR extraction (GPT-4o)
    extracted_text = call_poe_model(
        "gpt-4o",
        [{"role": "system", "content": "Extract all text from the attachments. Do NOT summarize."}],
        attachments=attachments
    )

    # Step 2: Content Analysis with GPT-5.1
    analysis_prompt = f"""
Analyze this medical content and return ONLY JSON as described.
Content:
{extracted_text}

User context: {user_context}

Return ONLY clean JSON.
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

    # Step 3: Flashcard Generation
    flashcard_prompt = f"""
SYSTEM PROMPT:
You are NursingAI — generate STRICTLY formatted medical flashcards.
Follow ALL rules the user provided previously.
Use extracted content and JSON analysis.

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
