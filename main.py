# main.py
import os
import io
import json
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
# from ai_model import call_openai_extract  # لو عايزة تحلي النصوص بطريقة OpenAI

# ---------- Load ENV ----------
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    raise Exception("GROQ_API_KEY missing")

groq_client = Groq(api_key=GROQ_KEY)

# ---------- App ----------
app = FastAPI(title="Voice & Text Finance Analyzer")

# ---------- CORS -----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # لو حابة تحددي دومين معين استبدلي هنا
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# ---------- Home ----------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------- Text Input ----------
class TextInput(BaseModel):
    text: str

FINANCE_PROMPT = """
حلل الجملة التالية من حيث البيانات المالية فقط.
ارجع JSON فقط بالشكل التالي:

{{
  "amount": <number|null>,
  "category": "<food|transport|shopping|bills|other>",
  "item": "<what was bought or paid for | null>",
  "place": "<optional>",
  "type": "<expense|income>"
}}

الجملة: "{text}"
"""

# ---------- Text Analyze ----------
@app.post("/analyze")
def analyze_text(input: TextInput):
    prompt = FINANCE_PROMPT.format(text=input.text)
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        output = response.choices[0].message.content
        start = output.find("{")
        end = output.rfind("}")
        parsed = json.loads(output[start:end+1])
        return {"analysis": parsed}
    except Exception as e:
        print("Text Analysis Error:", e)
        raise HTTPException(status_code=500, detail="Text analysis failed: " + str(e))

# ---------- Voice Analyze ----------
@app.post("/voice")
async def analyze_voice(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio")

    try:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = file.filename or "voice.webm"
        print(f"Processing audio file: {audio_file.name} ({len(audio_bytes)} bytes)")

        transcript = groq_client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=audio_file
        )
        text = transcript.text
        print("Transcription successful:", text)

    except Exception as e:
        print("STT Error:", e)
        raise HTTPException(status_code=500, detail="STT failed: " + str(e))

    # Text → Finance Analysis
    prompt = FINANCE_PROMPT.format(text=text)
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        output = response.choices[0].message.content
        start = output.find("{")
        end = output.rfind("}")
        parsed = json.loads(output[start:end+1])
        print("Analysis successful:", parsed)

        return {
            "text": text,
            "analysis": parsed
        }
    except Exception as e:
        print("Analysis Error:", e)
        raise HTTPException(status_code=500, detail="Analysis failed: " + str(e))

# ---------- Run Server ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
