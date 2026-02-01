# main.py
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydub import AudioSegment
from ai_model import analyze_text

app = FastAPI(title="Arabic Voice Finance Analyzer")

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# ---------- Home ----------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------- Text Analyze ----------
@app.post("/analyze")
def text_analyze(payload: dict):
    text = payload.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    analysis = analyze_text(text)
    return JSONResponse({"text": text, "analysis": analysis})

# ---------- Voice Analyze ----------
@app.post("/voice")
async def voice_analyze(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        audio_bytes = await file.read()
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_channels(1).set_frame_rate(16000)

        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)
        buffer.name = "voice.wav"

        # Groq transcription عربي
        from groq import Groq
        import os
        from dotenv import load_dotenv
        load_dotenv()
        GROQ_KEY = os.getenv("GROQ_API_KEY")
        client = Groq(api_key=GROQ_KEY)

        transcript = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=buffer,
            language="ar"
        )

        text = transcript.text.strip()
        analysis = analyze_text(text)

        return JSONResponse({"text": text, "analysis": analysis})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
