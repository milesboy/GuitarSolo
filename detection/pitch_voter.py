"""Pitch verification by CQT voting.

For each note detected by Basic Pitch, takes multiple CQT snapshots
during the note's active period and votes on the correct pitch.
BP is great at onset detection; CQT resolves pitch ambiguity.
"""
import librosa
import numpy as np


def build_cqt(y, sr, hop_length=512):
    """Build a high-resolution CQT for pitch voting."""
    y_harmonic, _ = librosa.effects.hpss(y)
    tuning = librosa.estimate_tuning(y=y_harmonic, sr=sr)
    fmin = librosa.note_to_hz("E2") * (2 ** (tuning / 12))

    n_semitones = 52  # E2 to G6
    bins_per_semi = 3
    cqt_raw = np.abs(librosa.cqt(
        y=y_harmonic, sr=sr, fmin=fmin, hop_length=hop_length,
        n_bins=n_semitones * bins_per_semi,
        bins_per_octave=12 * bins_per_semi,
    ))

    cqt = np.zeros((n_semitones, cqt_raw.shape[1]))
    for i in range(n_semitones):
        cqt[i] = np.max(cqt_raw[i * bins_per_semi:(i + 1) * bins_per_semi], axis=0)

    return cqt, hop_length, sr


def vote_pitch(cqt, sr, hop_length, onset_time, duration,
               detected_midi, n_samples=5, search_range=12):
    """Vote on the correct pitch for a single note.

    Takes n_samples CQT snapshots evenly spaced during the note's
    active period. Each snapshot votes for the strongest peak within
    search_range semitones of the detected pitch.

    Args:
        cqt: CQT magnitude array (n_semitones x n_frames)
        sr: sample rate
        hop_length: CQT hop length
        onset_time: note start time in seconds
        duration: note duration in seconds
        detected_midi: MIDI pitch from Basic Pitch
        n_samples: number of CQT snapshots to take
        search_range: semitones above/below to search

    Returns:
        (voted_midi, confidence, votes_detail)
        voted_midi: the winning MIDI pitch
        confidence: fraction of samples that agreed
        votes_detail: dict of {midi: vote_count}
    """
    n_semitones = cqt.shape[0]
    base_midi = 40  # E2 = bin 0

    # Convert detected MIDI to bin index
    detected_bin = detected_midi - base_midi
    search_low = max(0, detected_bin - search_range)
    search_high = min(n_semitones, detected_bin + search_range + 1)

    # Sample points: evenly spaced through the note's duration
    # Avoid the very start (attack transient) and very end (decay)
    sample_start = onset_time + duration * 0.1
    sample_end = onset_time + duration * 0.9
    if sample_end <= sample_start:
        sample_start = onset_time
        sample_end = onset_time + duration

    sample_times = np.linspace(sample_start, sample_end, n_samples)

    votes = {}
    window = max(1, int(0.03 * sr / hop_length))  # ~30ms analysis window

    for t in sample_times:
        frame = librosa.time_to_frames(t, sr=sr, hop_length=hop_length)
        end_frame = min(frame + window, cqt.shape[1])
        if frame >= cqt.shape[1]:
            continue

        # Average CQT energy in the search range
        mag = np.mean(cqt[search_low:search_high, frame:end_frame], axis=1)
        if len(mag) == 0 or np.max(mag) == 0:
            continue

        # Find the strongest peak (local maximum)
        best_bin = None
        best_mag = 0
        for b in range(len(mag)):
            actual_bin = search_low + b
            left = mag[b - 1] if b > 0 else 0
            right = mag[b + 1] if b < len(mag) - 1 else 0
            if mag[b] > left and mag[b] > right and mag[b] > best_mag:
                best_mag = mag[b]
                best_bin = actual_bin

        if best_bin is not None:
            voted_midi = best_bin + base_midi
            votes[voted_midi] = votes.get(voted_midi, 0) + 1

    if not votes:
        return detected_midi, 0.0, {}

    # Winner: most votes, break ties by proximity to detected pitch
    winner = max(votes.keys(),
                 key=lambda m: (votes[m], -abs(m - detected_midi)))
    confidence = votes[winner] / n_samples

    return winner, confidence, votes


def verify_notes(notes, y, sr, n_samples=5, min_confidence=0.4,
                 verbose=True):
    """Verify and correct pitches for all notes using CQT voting.

    Args:
        notes: list of (time, note_name, freq, velocity, duration)
        y: audio signal
        sr: sample rate
        n_samples: CQT snapshots per note
        min_confidence: minimum vote fraction to override BP pitch
        verbose: print progress

    Returns:
        corrected notes list, number of corrections
    """
    if verbose:
        print(f"  Building CQT for pitch voting...", end=" ", flush=True)

    cqt, hop, sr = build_cqt(y, sr)
    if verbose:
        print("done")

    corrected = []
    corrections = 0
    octave_fixes = 0

    for t, name, freq, vel, dur in notes:
        clean = name.replace("\u266f", "#").replace("\u266d", "b")
        try:
            detected_midi = librosa.note_to_midi(clean)
        except Exception:
            corrected.append((t, name, freq, vel, dur))
            continue

        # Skip very short notes (not enough samples)
        if dur < 0.05:
            corrected.append((t, name, freq, vel, dur))
            continue

        voted_midi, confidence, votes = vote_pitch(
            cqt, sr, hop, t, dur, detected_midi, n_samples=n_samples)

        if voted_midi != detected_midi and confidence >= min_confidence:
            # Correction!
            new_name = librosa.midi_to_note(voted_midi)
            new_freq = float(librosa.midi_to_hz(voted_midi))
            corrected.append((t, new_name, new_freq, vel, dur))
            corrections += 1

            diff = voted_midi - detected_midi
            if abs(diff) == 12:
                octave_fixes += 1
        else:
            corrected.append((t, name, freq, vel, dur))

    if verbose:
        print(f"  Pitch voting: {corrections} corrections "
              f"({octave_fixes} octave fixes) from {len(notes)} notes")

    return corrected, corrections
