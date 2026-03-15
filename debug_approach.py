"""
Test a new approach for note detection:

Key insights from the trace:
1. Spectral leakage creates phantom low notes (G#2 next to A2, F2/F#2/G2 as leakage)
2. The lowest bin is NOT always the bass — need to find the DOMINANT low note
3. D#3 appears when E3 is the real note (leakage -1 semitone)

New approach:
- Find bass as the strongest note in the low range, not just the lowest
- Use local peak picking to avoid spectral leakage (require a note to be a local max)
- Apply harmonic suppression only from confirmed real notes (peaks)
- Key-aware filtering: for notes in the key, be more lenient

Actually, the simplest fix: use LOCAL PEAK detection in the CQT magnitude.
A real note is a peak — higher than its neighbors. Spectral leakage bins
are NOT peaks — they rise monotonically toward the actual note.

Let's test this.
"""
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
for i in range(n_semitones):
    midi_note = librosa.hz_to_midi(fmin_standard) + i
    bin_notes.append(librosa.midi_to_note(midi_note))

onset_frames = librosa.onset.onset_detect(y=y_harmonic, sr=sr, hop_length=hop_length, backtrack=True)
onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

window_frames = max(1, int(0.25 * sr / hop_length))
all_mags = cqt_collapsed[cqt_collapsed > 0]
global_threshold = np.percentile(all_mags, 50)

harm_intervals = {12, 19, 24, 28, 31}

def is_local_peak(mag_window, idx, n_semitones):
    """Check if bin idx is a local peak (higher than both neighbors)."""
    left = mag_window[idx - 1] if idx > 0 else 0
    right = mag_window[idx + 1] if idx < n_semitones - 1 else 0
    return mag_window[idx] > left and mag_window[idx] > right

print("Testing LOCAL PEAK approach:")
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
        continue

    threshold = max(max_mag * 0.15, global_threshold)
    active = np.where(mag_window > threshold)[0]
    if len(active) == 0:
        continue

    # Filter to only local peaks
    peaks = [idx for idx in active if is_local_peak(mag_window, idx, n_semitones)]

    if not peaks:
        continue

    # Bass is the lowest PEAK
    bass_idx = peaks[0]

    # Build harmonics of bass
    bass_harmonics = set()
    for h in harm_intervals:
        for tol in [-1, 0, 1]:
            bass_harmonics.add(bass_idx + h + tol)

    # Keep peaks that aren't harmonics of bass
    real_bins = [bass_idx]
    for idx in peaks[1:]:
        if idx in bass_harmonics:
            continue
        real_bins.append(idx)

    # Suppress harmonics of each real note
    final_bins = []
    all_harmonics = set()
    for idx in sorted(real_bins):
        if idx in all_harmonics:
            continue
        final_bins.append(idx)
        for h in harm_intervals:
            for tol in [-1, 0, 1]:
                all_harmonics.add(idx + h + tol)

    note_names = [f"{bin_notes[idx]}({mag_window[idx]:.1f})" for idx in final_bins]
    peak_names = [f"{bin_notes[idx]}({mag_window[idx]:.1f})" for idx in peaks]
    print(f"t={onset_t:.3f}s  peaks={peak_names}  ->  final={note_names}")
