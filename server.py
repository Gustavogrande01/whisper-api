from fastapi import FastAPI, UploadFile
from faster_whisper import WhisperModel
import shutil

app = FastAPI()
model = WhisperModel("small")

@app.post("/transcribe")
async def transcribe(file: UploadFile):

    audio_path = "audio.wav"

    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    segments, info = model.transcribe(audio_path)

    text = ""
    for segment in segments:
        text += segment.text + " "

    return {"transcription": text}
