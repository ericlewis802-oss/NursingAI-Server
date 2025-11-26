import os
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

POE_ACCESS_KEY = os.environ.get("POE_ACCESS_KEY")
HEADERS = {"Authorization": f"Bearer {POE_ACCESS_KEY}"}

app = FastAPI()


# --------------------------------------------------------------------
# Call Poe Model Helper
# --------------------------------------------------------------------
def call_poe_model(model, messages):
    response = requests.post(
        f"https://api.poe.com/bot/{model}",
        headers=HEADERS,
        json={"messages": messages},
        timeout=120
    )
    response.raise_for_status()
    return response.json()["output_text"]


# --------------------------------------------------------------------
# Health Check
# --------------------------------------------------------------------
@app.get("/")
async def health():
    return {"status": "ok", "message": "NursingAI server running"}


# --------------------------------------------------------------------
# Main Webhook (Poe → Your Server)
# --------------------------------------------------------------------
@app.post("/poe_webhook")
async def poe_webhook(request: Request):
    data = await request.json()

    ############################
    # Get message content
    ############################
    text = data.get("message", "")
    attachments = data.get("attachments", [])
    messages = [{"role": "user", "content": text}]

    ############################
    # CASE 1: Attachment → OCR → Analyze → Flashcards
    ############################
    if attachments:
        # ---- 1) OCR with GPT‑4o ----
        ocr_text = call_poe_model(
            "gpt-4o",
            [{"role": "user", "content": f"OCR this: {attachments}"}]
        )

        # ---- 2) Analyze for nursing/medical content ----
        analysis_prompt = (
            "Analyze the following text and return JSON:\n"
            "1. medical_topics: list\n"
            "2. needs_web_search: true/false\n"
            f"Text:\n{ocr_text}"
        )
        analysis_raw = call_poe_model(
            "gpt-5.1",
            [{"role": "user", "content": analysis_prompt}]
        )

        # ---- 3) Generate flashcards ----
        fc_prompt = (
            "Generate strictly formatted medical flashcards.\n"
            f"TEXT:\n{ocr_text}\n\n"
            f"ANALYSIS JSON:\n{analysis_raw}"
        )
        flashcards = call_poe_model(
            "gpt-5.1",
            [{"role": "user", "content": fc_prompt}]
        )

        return JSONResponse({"reply": flashcards})

    ############################
    # CASE 2: No attachment → Normal chat
    ############################
    reply = call_poe_model("gpt-4o-mini", messages)

    return JSONResponse({"reply": reply})
