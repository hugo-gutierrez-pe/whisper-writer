#!/usr/bin/env python3
"""
Dictado por voz → portapapeles.
Enter para grabar, Enter para parar y transcribir.
"""
import subprocess
import sys
import tempfile
import os
from faster_whisper import WhisperModel

MODEL_SIZE = "medium"
LANGUAGE = "es"

print("Cargando modelo...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print(f"Modelo '{MODEL_SIZE}' listo.\n")

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
    segments, _ = model.transcribe(wav_path, language=LANGUAGE)
    texto = " ".join(s.text.strip() for s in segments)
    os.unlink(wav_path)

    if texto:
        subprocess.run(["wl-copy", texto])
        print(f"✓ Copiado: {texto}")
    else:
        print("(sin texto detectado)")
    print()
