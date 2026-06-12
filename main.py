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
        cookies_path = None

        # Write cookies from env variable if available
        cookies_content = os.environ.get("INSTAGRAM_COOKIES")
        if cookies_content:
            cookies_path = os.path.join(tmpdir, "cookies.txt")
            with open(cookies_path, "w") as f:
                f.write(cookies_content)

        cmd = ["yt-dlp", "-x", "--audio-format", "mp3", "-o", audio_path]
        if cookies_path:
            cmd += ["--cookies", cookies_path]
        cmd.append(req.url)

        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=400, detail=f"Download failed: {e.stderr.decode()}")
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=408, detail="Download timed out")

        if not os.path.exists(audio_path):
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
