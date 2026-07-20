#!/usr/bin/env python3
"""
Dictado por voz → portapapeles.
Enter para grabar, Enter para parar y transcribir.
"""
import subprocess
import sys
import tempfile
import os
import numpy as np
import wave
from faster_whisper import WhisperModel

MODEL_SIZE = "medium"
LANGUAGE = "es"
AUDIO_DEVICE = "plughw:2,0"

def tiene_voz(wav_path, umbral_rms=800):
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

print("Cargando modelo...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print(f"Modelo '{MODEL_SIZE}' listo.\n")

while True:
    input("[ Enter para GRABAR ] ")
    print("Grabando... (Enter para detener)")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    proc = subprocess.Popen(
        ["arecord", "-D", AUDIO_DEVICE, "-f", "cd", "-r", "16000", "-c", "1", wav_path],
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
    segments, _ = model.transcribe(wav_path, language=LANGUAGE, vad_filter=True)
    texto = " ".join(s.text.strip() for s in segments)
    os.unlink(wav_path)

    if texto:
        subprocess.run(["wl-copy", texto])
        print(f"✓ Copiado: {texto}")
    else:
        print("(sin texto detectado)")
    print()
