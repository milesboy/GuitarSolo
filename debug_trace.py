"""Trace through detect_notes logic step by step for the first few onsets to understand what's being filtered."""
import librosa
import numpy as np

WAV = r"C:\Users\Roy\CodingProjects\GuitarSolo\downloads\Kelly Valleau - In My Life (The Beatles) - Fingerstyle Guitar.wav"

y, sr = librosa.load(WAV, sr=None)
y_harmonic, _ = librosa.effects.hpss(y)
tuning_offset = librosa.estimate_tuning(y=y_harmonic, sr=sr)
fmin = librosa.note_to_hz("E2") * (2 ** (tuning_offset / 12))

hop_length = 512
n_semitones = 48
bins_per_semi = 3
cqt_raw = np.abs(librosa.cqt(
    y=y_harmonic, sr=sr, fmin=fmin, hop_length=hop_length,
    n_bins=n_semitones * bins_per_semi,
    bins_per_octave=12 * bins_per_semi,
))

cqt_collapsed = np.zeros((n_semitones, cqt_raw.shape[1]))
for i in range(n_semitones):
    cqt_collapsed[i] = np.max(cqt_raw[i * bins_per_semi:(i + 1) * bins_per_semi], axis=0)

fmin_standard = librosa.note_to_hz("E2")
bin_notes = []
bin_freqs = []
for i in range(n_semitones):
    midi_note = librosa.hz_to_midi(fmin_standard) + i
    bin_notes.append(librosa.midi_to_note(midi_note))
    bin_freqs.append(float(librosa.midi_to_hz(midi_note)))

onset_frames = librosa.onset.onset_detect(y=y_harmonic, sr=sr, hop_length=hop_length, backtrack=True)
onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

window_frames = max(1, int(0.25 * sr / hop_length))
all_mags = cqt_collapsed[cqt_collapsed > 0]
global_threshold = np.percentile(all_mags, 50)

harm_intervals = {12, 19, 24, 28, 31}

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
        continue

    active = np.where(mag_window > max(max_mag * 0.2, global_threshold))[0]
    if len(active) == 0:
        continue

    bass_idx = active[0]
    bass_harmonics = set()
    for h in harm_intervals:
        for tol in [-1, 0, 1]:
            bass_harmonics.add(bass_idx + h + tol)

    print(f"\nt={onset_t:.3f}s")
    print(f"  Active bins: {[(idx, bin_notes[idx], f'{mag_window[idx]:.2f}') for idx in active]}")
    print(f"  Bass: bin {bass_idx} = {bin_notes[bass_idx]}")
    print(f"  Bass harmonics (bins): {sorted(bass_harmonics)}")

    real_bins = [bass_idx]
    for idx in active:
        if idx == bass_idx:
            continue
        if idx in bass_harmonics:
            print(f"    REMOVED as bass harmonic: bin {idx} = {bin_notes[idx]}")
        else:
            print(f"    KEPT: bin {idx} = {bin_notes[idx]}")
            real_bins.append(idx)

    # Step 3: suppress harmonics of each real note
    final_bins = []
    all_harmonics = set()
    for idx in sorted(real_bins):
        if idx in all_harmonics:
            print(f"    REMOVED in step 3 (harmonic of lower real note): bin {idx} = {bin_notes[idx]}")
            continue
        final_bins.append(idx)
        for h in harm_intervals:
            for tol in [-1, 0, 1]:
                all_harmonics.add(idx + h + tol)

    print(f"  Final notes: {[(idx, bin_notes[idx]) for idx in final_bins]}")
