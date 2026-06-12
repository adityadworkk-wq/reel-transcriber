import os, tempfile, subprocess
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import openai

app = FastAPI()
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class ReelRequest(BaseModel):
    url: str

@app.post("/transcribe")
async def transcribe(req: ReelRequest):
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")
        try:
            subprocess.run(
                ["yt-dlp", "-x", "--audio-format", "mp3", "-o", audio_path, req.url],
                check=True, capture_output=True, timeout=120
            )
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=400, detail=f"Download failed: {e.stderr.decode()}")
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=408, detail="Download timed out")

        if not os.path.exists(audio_path):
            # yt-dlp sometimes appends extension
            matches = [f for f in os.listdir(tmpdir) if f.startswith("audio")]
            if not matches:
                raise HTTPException(status_code=400, detail="Audio file not found after download")
            audio_path = os.path.join(tmpdir, matches[0])

        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                task="translate"
            )

    return {"transcript": result.text}

app.mount("/", StaticFiles(directory="static", html=True), name="static")
