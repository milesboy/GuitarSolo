"""Run current analyze.py detect_notes and show what it produces for the first 4 bars."""
import librosa
import numpy as np
from analyze import detect_notes

WAV = r"C:\Users\Roy\CodingProjects\GuitarSolo\downloads\Kelly Valleau - In My Life (The Beatles) - Fingerstyle Guitar.wav"

print("Loading audio...")
y, sr = librosa.load(WAV, sr=None)

# Detect BPM
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
bpm = float(np.round(tempo[0] if hasattr(tempo, '__len__') else tempo))
print(f"BPM: {bpm}")

notes = detect_notes(y, sr, bpm)

# Show notes in first ~10 seconds
print(f"\nNotes detected in first 10s (BPM={bpm}):")
print(f"{'Time':>7s} {'Note':>5s} {'Vel':>4s}")
print("-" * 20)
for t, note, freq, vel in notes:
    if t > 10.5:
        break
    print(f"{t:7.3f} {note:>5s} {vel:>4d}")
