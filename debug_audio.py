"""Debug script: analyze the first ~10 seconds of In My Life to understand what notes are actually there."""
import librosa
import numpy as np

WAV = r"C:\Users\Roy\CodingProjects\GuitarSolo\downloads\Kelly Valleau - In My Life (The Beatles) - Fingerstyle Guitar.wav"

print("Loading audio...")
y, sr = librosa.load(WAV, sr=None)
duration = librosa.get_duration(y=y, sr=sr)
print(f"Sample rate: {sr}, Duration: {duration:.1f}s")

# Detect tempo
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
bpm = float(np.round(tempo[0] if hasattr(tempo, '__len__') else tempo))
print(f"Detected BPM: {bpm}")
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
print(f"First 20 beat times: {beat_times[:20]}")

# How long is 4 bars at the detected BPM?
bar_duration = 4 * 60.0 / bpm  # 4 beats per bar
four_bars = 4 * bar_duration
print(f"4 bars = {four_bars:.2f}s")

# Trim to first 12 seconds for analysis
analysis_duration = 12.0
y_short = y[:int(analysis_duration * sr)]

# Harmonic separation
y_harmonic, _ = librosa.effects.hpss(y_short)

# Tuning
tuning_offset = librosa.estimate_tuning(y=y_harmonic, sr=sr)
print(f"Tuning offset: {tuning_offset:.2f} semitones")
fmin = librosa.note_to_hz("E2") * (2 ** (tuning_offset / 12))

# Onset detection
hop_length = 512
onset_frames = librosa.onset.onset_detect(y=y_harmonic, sr=sr, hop_length=hop_length, backtrack=True)
onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
print(f"\nOnsets in first {analysis_duration}s: {len(onset_times)}")
for i, t in enumerate(onset_times):
    print(f"  Onset {i:2d}: {t:.3f}s")

# High-res CQT
n_semitones = 48
bins_per_semi = 3
cqt_raw = np.abs(librosa.cqt(
    y=y_harmonic, sr=sr, fmin=fmin, hop_length=hop_length,
    n_bins=n_semitones * bins_per_semi,
    bins_per_octave=12 * bins_per_semi,
))

# Collapse to 1 bin per semitone
cqt_collapsed = np.zeros((n_semitones, cqt_raw.shape[1]))
for i in range(n_semitones):
    cqt_collapsed[i] = np.max(cqt_raw[i * bins_per_semi:(i + 1) * bins_per_semi], axis=0)

# Note names for each bin
fmin_standard = librosa.note_to_hz("E2")
bin_notes = []
bin_midis = []
for i in range(n_semitones):
    midi_note = librosa.hz_to_midi(fmin_standard) + i
    bin_notes.append(librosa.midi_to_note(midi_note))
    bin_midis.append(midi_note)

print(f"\nCQT range: {bin_notes[0]} to {bin_notes[-1]}")

# For each onset, show the top energy bins
window_frames = max(1, int(0.25 * sr / hop_length))
all_mags = cqt_collapsed[cqt_collapsed > 0]
global_threshold = np.percentile(all_mags, 50)

print(f"\nGlobal threshold: {global_threshold:.4f}")

print("\n" + "=" * 80)
print("DETAILED NOTE ANALYSIS PER ONSET (first 10s)")
print("=" * 80)

for onset_t in onset_times:
    if onset_t > 10.0:
        break
    onset_frame = librosa.time_to_frames(onset_t, sr=sr, hop_length=hop_length)
    start = onset_frame
    end = min(start + window_frames, cqt_collapsed.shape[1])
    if start >= cqt_collapsed.shape[1]:
        continue

    mag_window = np.mean(cqt_collapsed[:, start:end], axis=1)
    max_mag = np.max(mag_window)
    if max_mag < global_threshold:
        print(f"\n  t={onset_t:.3f}s -- below threshold")
        continue

    # Show all active bins
    active = np.where(mag_window > max(max_mag * 0.15, global_threshold))[0]

    print(f"\n  t={onset_t:.3f}s  (max_mag={max_mag:.4f})")
    print(f"  {'Bin':>4s} {'Note':>5s} {'MIDI':>5s} {'Magnitude':>10s} {'Rel%':>6s}")
    print(f"  {'-'*35}")
    for idx in active:
        rel = mag_window[idx] / max_mag * 100
        print(f"  {idx:4d} {bin_notes[idx]:>5s} {bin_midis[idx]:>5.0f} {mag_window[idx]:10.4f} {rel:5.1f}%")


# Also do chroma analysis for the first 10 seconds
print("\n" + "=" * 80)
print("CHROMA ANALYSIS (first 10s)")
print("=" * 80)

chroma = librosa.feature.chroma_cqt(y=y_short, sr=sr, hop_length=hop_length)
times = librosa.frames_to_time(range(chroma.shape[1]), sr=sr, hop_length=hop_length)

# Show chroma at each onset
CHROMA_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
for onset_t in onset_times:
    if onset_t > 10.0:
        break
    frame = librosa.time_to_frames(onset_t, sr=sr, hop_length=hop_length)
    end = min(frame + 5, chroma.shape[1])
    if frame >= chroma.shape[1]:
        continue
    chroma_slice = np.mean(chroma[:, frame:end], axis=1)
    top_indices = np.argsort(chroma_slice)[::-1][:5]
    top_notes = [(CHROMA_NOTES[i], chroma_slice[i]) for i in top_indices]
    print(f"  t={onset_t:.3f}s: {', '.join(f'{n}={v:.3f}' for n, v in top_notes)}")
