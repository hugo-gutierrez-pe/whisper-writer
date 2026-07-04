#!/usr/bin/env python3
"""
Dictado por voz → portapapeles usando Groq Whisper API.
Enter para grabar, Enter para parar y transcribir.
"""
import subprocess
import sys
import tempfile
import os
from pathlib import Path
from groq import Groq

API_KEY_PATH = Path.home() / "Documentos" / "groq_api.txt"
LANGUAGE = "es"
MODEL = "whisper-large-v3-turbo"

api_key = API_KEY_PATH.read_text().strip()
client = Groq(api_key=api_key)
print(f"Groq Whisper listo ({MODEL}).\n")

while True:
    input("[ Enter para GRABAR ] ")
    print("Grabando... (Enter para detener)")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    proc = subprocess.Popen(
        ["arecord", "-f", "cd", "-r", "16000", "-c", "1", wav_path],
        stderr=subprocess.DEVNULL,
    )

    input("")
    proc.terminate()
    proc.wait()

    print("Transcribiendo...", end=" ", flush=True)
    with open(wav_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(
            file=("audio.wav", audio_file),
            model=MODEL,
            language=LANGUAGE,
        )
    os.unlink(wav_path)

    texto = result.text.strip()
    if texto:
        subprocess.run(["wl-copy", texto])
        print(f"✓ Copiado: {texto}")
    else:
        print("(sin texto detectado)")
    print()
