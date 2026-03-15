"""Analyze the first ~10 seconds of the wav file to understand what notes are actually being played."""
import librosa
import numpy as np

wav_path = r"C:\Users\Roy\CodingProjects\GuitarSolo\downloads\Kelly Valleau - In My Life (The Beatles) - Fingerstyle Guitar.wav"

print("Loading audio...")
y, sr = librosa.load(wav_path, sr=None)
duration = librosa.get_duration(y=y, sr=sr)
print(f"Full duration: {duration:.1f}s, sr={sr}")

# Detect tempo
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
bpm = float(np.round(tempo[0] if hasattr(tempo, '__len__') else tempo))
print(f"Detected BPM: {bpm}")

beat_times = librosa.frames_to_time(beat_frames, sr=sr)
print(f"First 20 beat times: {beat_times[:20]}")
sec_per_beat = 60.0 / bpm
print(f"Seconds per beat: {sec_per_beat:.3f}")
print(f"4 bars = 16 beats = {16 * sec_per_beat:.1f}s")

# Focus on first ~10 seconds
y_clip = y[:int(10 * sr)]

# Harmonic separation
y_harmonic, _ = librosa.effects.hpss(y_clip)

# Estimate tuning
tuning_offset = librosa.estimate_tuning(y=y_harmonic, sr=sr)
print(f"\nTuning offset: {tuning_offset:.3f} semitones")

# CQT analysis
fmin = librosa.note_to_hz("E2") * (2 ** (tuning_offset / 12))
hop_length = 512
n_semitones = 48
bins_per_semi = 3

cqt_raw = np.abs(librosa.cqt(
    y=y_harmonic, sr=sr, fmin=fmin, hop_length=hop_length,
    n_bins=n_semitones * bins_per_semi,
    bins_per_octave=12 * bins_per_semi,
))

# Collapse to semitones
cqt = np.zeros((n_semitones, cqt_raw.shape[1]))
for i in range(n_semitones):
    cqt[i] = np.max(cqt_raw[i * bins_per_semi:(i + 1) * bins_per_semi], axis=0)

# Build bin names
fmin_standard = librosa.note_to_hz("E2")
bin_notes = []
bin_midi = []
for i in range(n_semitones):
    midi_note = librosa.hz_to_midi(fmin_standard) + i
    bin_notes.append(librosa.midi_to_note(midi_note))
    bin_midi.append(int(midi_note))

print(f"\nCQT bins: {bin_notes[0]} to {bin_notes[-1]}")
print(f"MIDI range: {bin_midi[0]} to {bin_midi[-1]}")

# Onset detection
onset_frames = librosa.onset.onset_detect(
    y=y_harmonic, sr=sr, hop_length=hop_length, backtrack=True,
)
onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

print(f"\nOnsets in first 10s ({len(onset_times)} total):")
for i, t in enumerate(onset_times):
    if t > 10:
        break

    frame = onset_frames[i]
    window_frames = max(1, int(0.25 * sr / hop_length))
    start = frame
    end = min(start + window_frames, cqt.shape[1])

    if start >= cqt.shape[1]:
        continue

    mag = np.mean(cqt[:, start:end], axis=1)
    max_mag = np.max(mag)

    # Find active bins
    threshold = max(max_mag * 0.15, np.percentile(cqt[cqt > 0], 40) if np.any(cqt > 0) else 0)
    active = np.where(mag > threshold)[0]

    if len(active) == 0:
        continue

    # Print all active bins sorted by magnitude
    active_sorted = sorted(active, key=lambda x: mag[x], reverse=True)
    notes_str = []
    for idx in active_sorted[:12]:
        pct = mag[idx] / max_mag * 100
        notes_str.append(f"{bin_notes[idx]}({pct:.0f}%)")

    print(f"  t={t:.3f}s: {', '.join(notes_str)}")

# Also look at chroma
print("\n\n--- Chroma analysis per onset (first 10s) ---")
chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=hop_length)
chroma_notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

for i, t in enumerate(onset_times):
    if t > 10:
        break

    frame = onset_frames[i]
    window_frames = max(1, int(0.15 * sr / hop_length))
    start = frame
    end = min(start + window_frames, chroma.shape[1])

    if start >= chroma.shape[1]:
        continue

    chroma_avg = np.mean(chroma[:, start:end], axis=1)
    max_c = np.max(chroma_avg)
    if max_c == 0:
        continue

    active_chroma = [(chroma_notes[j], chroma_avg[j] / max_c * 100) for j in range(12) if chroma_avg[j] > max_c * 0.3]
    active_chroma.sort(key=lambda x: x[1], reverse=True)

    chroma_str = ', '.join(f"{n}({p:.0f}%)" for n, p in active_chroma)
    print(f"  t={t:.3f}s: {chroma_str}")
