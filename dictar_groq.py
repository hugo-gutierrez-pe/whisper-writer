#!/usr/bin/env python3
"""
Dictado por voz → portapapeles usando Groq Whisper API.
Enter para grabar, Enter para parar y transcribir.
"""
import subprocess
import sys
import tempfile
import os
import numpy as np
import wave
from pathlib import Path
from groq import Groq

API_KEY_PATH = Path.home() / "Documentos" / "groq_api.txt"
LANGUAGE = "es"
MODEL = "whisper-large-v3-turbo"


def tiene_voz(wav_path, umbral_rms=250):
    with wave.open(wav_path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        return rms >= umbral_rms, rms


subprocess.run(
    ["amixer", "-c", "2", "cset", "numid=26", "3"],
    stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
)
subprocess.run(
    ["amixer", "-c", "2", "cset", "numid=21", "63"],
    stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
)

api_key = API_KEY_PATH.read_text().strip()
client = Groq(api_key=api_key)
print(f"Groq Whisper listo ({MODEL}).\n")

while True:
    input("[ Enter para GRABAR ] ")
    print("Grabando... (Enter para detener)")

    wav_path = os.path.join(tempfile.gettempdir(), "dictar_groq.wav")

    proc = subprocess.Popen(
        ["pw-record", "--target=49", "--rate=16000", "--channels=1", wav_path],
        stderr=subprocess.DEVNULL,
    )

    input("")
    proc.terminate()
    proc.wait()

    hay_voz, rms = tiene_voz(wav_path)
    if not hay_voz:
        print(f"(sin voz detectada, RMS={rms:.0f})")
        os.unlink(wav_path)
        print()
        continue

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
