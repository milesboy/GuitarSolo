"""Post-processor for Basic Pitch output.

Applies guitar-specific corrections to improve transcription accuracy:
1. Octave correction — snap notes to guitar range
2. False positive filtering — remove notes outside guitar range, merge duplicates
3. Duration improvement — use CQT sustain measurement for better durations
"""
import librosa
import numpy as np

# Guitar range: E2 (MIDI 40) to E6 (MIDI 88, highest harmonic)
GUITAR_MIDI_LOW = 40   # E2
GUITAR_MIDI_HIGH = 88  # E6
# Practical fretted range (no harmonics)
GUITAR_FRETTED_HIGH = 84  # C6 (24th fret on high E)


def fix_octave_errors(notes):
    """Snap notes to the nearest guitar-valid octave.

    Basic Pitch sometimes outputs notes an octave too high or low.
    If a note is outside guitar range, shift it by octaves until
    it fits. If it can't fit, remove it.

    Args:
        notes: list of (time, note_name, freq, velocity, duration)
    Returns:
        corrected notes list
    """
    corrected = []
    octave_fixes = 0

    for t, name, freq, vel, dur in notes:
        clean = name.replace("\u266f", "#").replace("\u266d", "b")
        try:
            midi = librosa.note_to_midi(clean)
        except Exception:
            corrected.append((t, name, freq, vel, dur))
            continue

        original_midi = midi

        # Shift into guitar range by octaves
        while midi < GUITAR_MIDI_LOW and midi + 12 <= GUITAR_MIDI_HIGH:
            midi += 12
        while midi > GUITAR_MIDI_HIGH and midi - 12 >= GUITAR_MIDI_LOW:
            midi -= 12

        if GUITAR_MIDI_LOW <= midi <= GUITAR_MIDI_HIGH:
            if midi != original_midi:
                octave_fixes += 1
                name = librosa.midi_to_note(midi)
                freq = float(librosa.midi_to_hz(midi))
            corrected.append((t, name, freq, vel, dur))
        # else: note is unreachable on guitar, drop it

    return corrected, octave_fixes


def filter_false_positives(notes, min_velocity=0.15, min_duration=0.05,
                           merge_window=0.03):
    """Remove spurious notes and merge near-duplicates.

    Args:
        notes: list of (time, note_name, freq, velocity, duration)
        min_velocity: minimum velocity to keep (0-1 scale from BP)
        min_duration: minimum duration in seconds
        merge_window: merge same-pitch notes within this time window
    Returns:
        filtered notes list, count removed
    """
    # Filter by velocity and duration
    filtered = []
    for n in notes:
        t, name, freq, vel, dur = n
        # BP velocity is 0-127, but we stored it as int
        # Low velocity notes are often artifacts
        if vel < min_velocity * 127:
            continue
        if dur < min_duration:
            continue
        filtered.append(n)

    removed = len(notes) - len(filtered)

    # Merge near-duplicates: same pitch within merge_window
    if not filtered:
        return filtered, removed

    filtered.sort(key=lambda n: (n[1], n[0]))  # sort by note name, then time
    merged = []
    i = 0
    while i < len(filtered):
        t, name, freq, vel, dur = filtered[i]
        # Look ahead for same note within merge window
        j = i + 1
        while j < len(filtered):
            t2, name2, freq2, vel2, dur2 = filtered[j]
            if name2 != name:
                break
            if t2 - t <= merge_window:
                # Merge: keep earlier onset, longer duration, higher velocity
                dur = max(dur, (t2 - t) + dur2)
                vel = max(vel, vel2)
                removed += 1
                j += 1
            else:
                break
        merged.append((t, name, freq, vel, dur))
        i = j

    # Re-sort by time
    merged.sort(key=lambda n: n[0])
    return merged, removed


def improve_durations_cqt(notes, y, sr, hop_length=512):
    """Use CQT sustain analysis to improve Basic Pitch durations.

    BP durations are often too short or too long. CQT energy tracking
    gives a more accurate picture of when each note actually stops.

    Only adjusts durations — doesn't change pitch or onset time.
    """
    if not notes:
        return notes

    y_harmonic, _ = librosa.effects.hpss(y)
    tuning = librosa.estimate_tuning(y=y_harmonic, sr=sr)
    fmin = librosa.note_to_hz("E2") * (2 ** (tuning / 12))

    n_semi = 52
    bins_per_semi = 3
    cqt_raw = np.abs(librosa.cqt(
        y=y_harmonic, sr=sr, fmin=fmin, hop_length=hop_length,
        n_bins=n_semi * bins_per_semi, bins_per_octave=12 * bins_per_semi))
    cqt = np.zeros((n_semi, cqt_raw.shape[1]))
    for i in range(n_semi):
        cqt[i] = np.max(cqt_raw[i*bins_per_semi:(i+1)*bins_per_semi], axis=0)

    fmin_std = librosa.note_to_hz("E2")
    window = max(1, int(0.1 * sr / hop_length))
    check_interval = int(0.1 * sr / hop_length)
    sustain_ratio = 0.20

    corrected = []
    dur_fixes = 0

    for t, name, freq, vel, dur in notes:
        clean = name.replace("\u266f", "#").replace("\u266d", "b")
        try:
            midi = librosa.note_to_midi(clean)
        except Exception:
            corrected.append((t, name, freq, vel, dur))
            continue

        note_bin = midi - 40  # E2 = MIDI 40 = bin 0
        if note_bin < 0 or note_bin >= n_semi:
            corrected.append((t, name, freq, vel, dur))
            continue

        # Measure actual sustain from CQT
        onset_frame = librosa.time_to_frames(t, sr=sr, hop_length=hop_length)
        end_frame = min(onset_frame + window, cqt.shape[1])
        if onset_frame >= cqt.shape[1]:
            corrected.append((t, name, freq, vel, dur))
            continue

        onset_mag = float(np.mean(cqt[note_bin, onset_frame:end_frame]))
        if onset_mag <= 0:
            corrected.append((t, name, freq, vel, dur))
            continue

        # Scan forward
        actual_dur = 0.1
        check_frame = onset_frame + check_interval
        while check_frame < cqt.shape[1]:
            c_end = min(check_frame + window, cqt.shape[1])
            energy = float(np.mean(cqt[note_bin, check_frame:c_end]))
            if energy < onset_mag * sustain_ratio:
                break
            actual_dur += 0.1
            check_frame += check_interval
            if actual_dur > 4.0:
                break

        # Only correct if significantly different (>40% off)
        if dur > 0 and abs(actual_dur - dur) / dur > 0.4:
            # Blend: trust BP for short notes, CQT for long sustains
            if actual_dur > dur:
                # CQT says longer — extend (BP often cuts short)
                new_dur = actual_dur
            else:
                # CQT says shorter — use average (BP sometimes too long)
                new_dur = (dur + actual_dur) / 2
            new_dur = max(new_dur, 0.05)
            if abs(new_dur - dur) / dur > 0.2:
                dur_fixes += 1
            dur = new_dur

        corrected.append((t, name, freq, vel, dur))

    return corrected, dur_fixes


def postprocess_bp(notes, y=None, sr=None, verbose=True):
    """Full post-processing pipeline for Basic Pitch output.

    Args:
        notes: list of (time, note_name, freq, velocity, duration)
        y, sr: audio signal (optional, needed for duration improvement)
        verbose: print progress

    Returns:
        corrected notes list
    """
    if verbose:
        print(f"\n=== BP Post-Processing ===")
        print(f"  Input: {len(notes)} notes")

    # Step 1: Fix octave errors
    notes, octave_fixes = fix_octave_errors(notes)
    if verbose:
        print(f"  Octave fixes: {octave_fixes}")

    # Step 2: Filter false positives
    notes, removed = filter_false_positives(notes)
    if verbose:
        print(f"  Filtered: {removed} removed, {len(notes)} remaining")

    # Step 3: CQT duration correction disabled — BP durations score
    # better on GuitarSet (0.579 full F1 vs 0.289 with CQT correction).
    # BP's neural network duration estimates are more accurate than
    # our CQT energy threshold approach.
    # TODO: revisit with a smarter duration model

    if verbose:
        print(f"  Output: {len(notes)} notes")

    return notes
