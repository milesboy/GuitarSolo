"""Note-level refinement pass.

After initial detection, walks through the audio at 64th-note resolution
to find missed notes, verify pitch accuracy, and correct durations.

This is the fine-grained counterpart to the parameter grid search:
- Grid search finds the best overall detection settings
- Note refiner catches individual notes that were missed or mis-detected
"""
import librosa
import numpy as np


def _build_cqt(y, sr, hop_length=256):
    """Build a high-resolution CQT for note verification."""
    y_harmonic, _ = librosa.effects.hpss(y)
    tuning_offset = librosa.estimate_tuning(y=y_harmonic, sr=sr)
    fmin = librosa.note_to_hz("E2") * (2 ** (tuning_offset / 12))

    n_semitones = 48
    bins_per_semi = 3
    cqt_raw = np.abs(librosa.cqt(
        y=y_harmonic, sr=sr, fmin=fmin, hop_length=hop_length,
        n_bins=n_semitones * bins_per_semi,
        bins_per_octave=12 * bins_per_semi,
    ))

    cqt_semitone = np.zeros((n_semitones, cqt_raw.shape[1]))
    for i in range(n_semitones):
        cqt_semitone[i] = np.max(
            cqt_raw[i * bins_per_semi:(i + 1) * bins_per_semi], axis=0)

    fmin_standard = librosa.note_to_hz("E2")
    bin_notes = []
    bin_freqs = []
    for i in range(n_semitones):
        midi_note = librosa.hz_to_midi(fmin_standard) + i
        bin_notes.append(librosa.midi_to_note(midi_note))
        bin_freqs.append(float(librosa.midi_to_hz(midi_note)))

    return cqt_semitone, bin_notes, bin_freqs, hop_length, y_harmonic


def find_missing_notes(y, sr, notes, bpm, verbose=True):
    """Scan for notes that the detector missed.

    Walks through every 64th-note grid position and checks CQT energy.
    If there's significant energy at a position with no detected note,
    identifies the pitch and adds it.

    Args:
        y: audio signal
        sr: sample rate
        notes: list of (time, note_name, freq, velocity, duration)
        bpm: tempo

    Returns:
        list of new notes to add (same format as input notes)
    """
    hop_length = 256
    cqt, bin_notes, bin_freqs, _, y_harm = _build_cqt(y, sr, hop_length)

    n_semitones = cqt.shape[0]
    sec_per_64th = 60.0 / bpm / 16  # duration of one 64th note

    # Build a set of onset slots (not sustain) — only the attack point
    # of each note is "occupied".  This allows finding new notes that
    # fall within another note's sustain (re-plucks, new voices, etc.)
    occupied_onsets = set()
    for t, note, freq, vel, dur in notes:
        onset_slot = round(t / sec_per_64th)
        # Mark a small window around each onset (±1 slot)
        for s in range(onset_slot - 1, onset_slot + 2):
            occupied_onsets.add(s)

    # Global energy threshold — 60th percentile (moderate bar)
    all_mags = cqt[cqt > 0]
    if len(all_mags) == 0:
        return []
    threshold = np.percentile(all_mags, 60)

    # Onset detection for candidate positions
    onset_frames = librosa.onset.onset_detect(
        y=y_harm, sr=sr, hop_length=hop_length, backtrack=True)
    onset_times = librosa.frames_to_time(
        onset_frames, sr=sr, hop_length=hop_length)

    window_frames = max(1, int(0.15 * sr / hop_length))  # 150ms window

    new_notes = []

    for onset_t in onset_times:
        slot = int(onset_t / sec_per_64th)
        # Skip if this slot already has a note
        if slot in occupied_onsets:
            continue

        onset_frame = librosa.time_to_frames(
            onset_t, sr=sr, hop_length=hop_length)
        start = onset_frame
        end = min(start + window_frames, cqt.shape[1])
        if start >= cqt.shape[1]:
            continue

        mag_window = np.mean(cqt[:, start:end], axis=1)
        max_mag = np.max(mag_window)
        if max_mag < threshold:
            continue

        # Find the strongest peak
        best_bin = np.argmax(mag_window)

        # Verify it's a local peak (not spectral leakage)
        left = mag_window[best_bin - 1] if best_bin > 0 else 0
        right = mag_window[best_bin + 1] if best_bin < n_semitones - 1 else 0
        if not (mag_window[best_bin] > left and mag_window[best_bin] > right):
            continue

        note_name = bin_notes[best_bin]
        freq = bin_freqs[best_bin]
        mag = float(mag_window[best_bin])

        # Estimate velocity (normalize against global range)
        vel = int(min(127, max(50, 50 + 77 * (mag / max_mag) ** 0.35)))

        # Estimate duration: scan forward until energy drops
        check_interval = int(0.1 * sr / hop_length)  # 100ms steps
        dur = 0.15  # minimum
        check_frame = start + check_interval
        while check_frame < cqt.shape[1]:
            c_end = min(check_frame + window_frames, cqt.shape[1])
            energy = np.mean(cqt[best_bin, check_frame:c_end])
            if energy < mag * 0.2:
                break
            dur += 0.1
            check_frame += check_interval
            if dur > 4.0:
                break

        new_notes.append((onset_t, note_name, freq, vel, dur))
        # Mark this onset as occupied
        for s in range(slot - 1, slot + 2):
            occupied_onsets.add(s)

    if verbose and new_notes:
        print(f"  Found {len(new_notes)} missing notes")

    return new_notes


def verify_notes(y, sr, notes, verbose=True):
    """Verify pitch and duration of each detected note against the CQT.

    Checks:
    1. Is the detected pitch the strongest CQT bin at that time?
    2. Does the note's duration match the actual sustain in the audio?

    Args:
        y: audio signal
        sr: sample rate
        notes: list of (time, note_name, freq, velocity, duration)

    Returns:
        corrected_notes: list with pitch/duration corrections applied
        corrections: list of (index, field, old_value, new_value)
    """
    hop_length = 256
    cqt, bin_notes, bin_freqs, _, _ = _build_cqt(y, sr, hop_length)
    n_semitones = cqt.shape[0]
    window_frames = max(1, int(0.15 * sr / hop_length))

    corrected = []
    corrections = []

    for i, (t, note_name, freq, vel, dur) in enumerate(notes):
        onset_frame = librosa.time_to_frames(t, sr=sr, hop_length=hop_length)
        start = onset_frame
        end = min(start + window_frames, cqt.shape[1])
        if start >= cqt.shape[1]:
            corrected.append((t, note_name, freq, vel, dur))
            continue

        mag_window = np.mean(cqt[:, start:end], axis=1)

        # --- Pitch verification ---
        # Find what bin this note should be
        if note_name in bin_notes:
            expected_bin = bin_notes.index(note_name)
        else:
            corrected.append((t, note_name, freq, vel, dur))
            continue

        # Check ±1 semitone for a stronger peak
        best_bin = expected_bin
        best_mag = mag_window[expected_bin]
        for offset in [-1, 1]:
            check_bin = expected_bin + offset
            if 0 <= check_bin < n_semitones:
                if mag_window[check_bin] > best_mag * 1.3:
                    best_bin = check_bin
                    best_mag = mag_window[check_bin]

        if best_bin != expected_bin:
            new_note = bin_notes[best_bin]
            new_freq = bin_freqs[best_bin]
            corrections.append((i, "pitch", note_name, new_note))
            note_name = new_note
            freq = new_freq

        # --- Duration verification ---
        # Scan forward to find actual sustain end
        check_interval = int(0.1 * sr / hop_length)
        onset_mag = mag_window[best_bin]
        actual_dur = 0.15
        check_frame = start + check_interval
        while check_frame < cqt.shape[1]:
            c_end = min(check_frame + window_frames, cqt.shape[1])
            energy = np.mean(cqt[best_bin, check_frame:c_end])
            if energy < onset_mag * 0.2:
                break
            actual_dur += 0.1
            check_frame += check_interval
            if actual_dur > 4.0:
                break

        # Only correct if significantly different (>30% off)
        if abs(actual_dur - dur) / max(dur, 0.01) > 0.3:
            corrections.append((i, "duration", f"{dur:.2f}", f"{actual_dur:.2f}"))
            dur = actual_dur

        corrected.append((t, note_name, freq, vel, dur))

    if verbose:
        print(f"  Verified {len(notes)} notes, "
              f"{len(corrections)} corrections")
        for idx, field, old, new in corrections[:10]:
            print(f"    note {idx}: {field} {old} -> {new}")
        if len(corrections) > 10:
            print(f"    ... and {len(corrections) - 10} more")

    return corrected, corrections


def refine_notes(y, sr, notes, bpm, verbose=True):
    """Full refinement pass: verify existing notes, then find missing ones.

    Returns:
        refined_notes: corrected + newly found notes, sorted by time
    """
    if verbose:
        print("\n=== Note Refinement Pass ===")
        print(f"  Input: {len(notes)} notes")

    # Step 1: verify pitch and duration of existing notes
    verified, corrections = verify_notes(y, sr, notes, verbose=verbose)

    # Step 2: find missing notes
    missing = find_missing_notes(y, sr, verified, bpm, verbose=verbose)

    # Merge and sort by time
    refined = sorted(verified + missing, key=lambda x: x[0])

    if verbose:
        print(f"  Output: {len(refined)} notes "
              f"({len(corrections)} corrected, {len(missing)} added)")

    return refined
