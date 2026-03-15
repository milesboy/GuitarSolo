"""Test detect_notes output for first 4 bars and regenerate MIDI."""
import librosa
import numpy as np
import sys
import os

# Force reimport of analyze module
if 'analyze' in sys.modules:
    del sys.modules['analyze']

from analyze import detect_notes, detect_tempo, detect_chords, export_midi

WAV = r"C:\Users\Roy\CodingProjects\GuitarSolo\downloads\Kelly Valleau - In My Life (The Beatles) - Fingerstyle Guitar.wav"

print("Loading audio...")
y, sr = librosa.load(WAV, sr=None)

# Detect BPM
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
bpm = float(np.round(tempo[0] if hasattr(tempo, '__len__') else tempo))
print(f"BPM: {bpm}")

sec_per_beat = 60.0 / bpm
bar_dur = 4 * sec_per_beat
print(f"Beat duration: {sec_per_beat:.3f}s, Bar duration: {bar_dur:.3f}s")

# Detect notes
print("\nDetecting notes...")
notes = detect_notes(y, sr, bpm)

# Expected notes (from debug_expected.py):
# Bar 1 (starts ~1.65s):
#   Beat 1 (1.653): A2 + C#4
#   Beat 2 (2.219): A2 + C#4
#   Beat 3 (2.795): A2 + A3 + C#4
#   Beat 3.5 (3.061): C#3 + D4
#   Beat 4 (3.371): D3 + E4
# Bar 2:
#   Beat 1-ish (3.648): E3 + E4 + G#4
#   Then G#4 sustains through rest of bar 2

print(f"\n{'='*60}")
print(f"DETECTED NOTES (first ~11s = 4 bars)")
print(f"{'='*60}")
print(f"{'Time':>7s} {'Note':>5s} {'Freq':>7s} {'Vel':>4s}")
print(f"{'-'*30}")

# Show notes in first 11 seconds
for t, note, freq, vel in notes:
    if t > 11.0:
        break
    print(f"{t:7.3f} {note:>5s} {freq:7.1f} {vel:>4d}")

print(f"\n{'='*60}")
print("EXPECTED NOTES:")
print("  t~1.65: A2 + C#4")
print("  t~2.22: A2 + C#4")
print("  t~2.80: A2 + A3 + C#4")
print("  t~3.06: C#3 + D4")
print("  t~3.37: D3 + E4")
print("  t~3.65: E3 + E4 + G#4")
print("  t~3.65-6.1: G#4 sustains")
print("  (Bars 3-4 repeat bars 1-2, starting ~6.1s)")
print(f"{'='*60}")

# Regenerate MIDI
print("\nRegenerating MIDI...")
chords = detect_chords(y, sr)
midi_path = export_midi(WAV, bpm, notes, chords)
print(f"MIDI saved to: {midi_path}")
