"""MIDI post-processing: quantize, filter, snap to key, merge."""
import librosa
import numpy as np


CHROMA_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Scale patterns (semitone intervals from root)
SCALE_PATTERNS = {
    "major": {0, 2, 4, 5, 7, 9, 11},
    "minor": {0, 2, 3, 5, 7, 8, 10},
}


def quantize_to_grid(notes, bpm, subdivision=16):
    """Snap note onsets to the nearest rhythmic grid position.

    Args:
        notes: list of (time, note, freq, velocity, duration)
        bpm: tempo
        subdivision: grid resolution (16 = 16th notes)

    Returns:
        New note list with quantized onset times.
    """
    grid_interval = 60.0 / bpm / (subdivision / 4)
    quantized = []
    for t, note, freq, vel, dur in notes:
        snapped = round(t / grid_interval) * grid_interval
        quantized.append((snapped, note, freq, vel, dur))
    return quantized


def filter_outliers(notes, min_duration_ms=30, min_velocity=20):
    """Remove implausibly short or quiet notes."""
    return [
        (t, note, freq, vel, dur) for t, note, freq, vel, dur in notes
        if dur * 1000 >= min_duration_ms and vel >= min_velocity
    ]


def snap_to_key(notes, key):
    """Shift off-key notes to nearest in-key pitch.

    Args:
        key: string like "A major" or "C# minor"
    """
    parts = key.split()
    if len(parts) < 2:
        return notes

    root_name = parts[0]
    scale_type = parts[1]

    if root_name not in CHROMA_NOTES or scale_type not in SCALE_PATTERNS:
        return notes

    root = CHROMA_NOTES.index(root_name)
    scale = SCALE_PATTERNS[scale_type]
    in_key_pcs = {(root + s) % 12 for s in scale}

    result = []
    for t, note, freq, vel, dur in notes:
        try:
            midi = librosa.note_to_midi(note.replace("♯", "#").replace("♭", "b"))
        except Exception:
            result.append((t, note, freq, vel, dur))
            continue

        pc = midi % 12
        if pc in in_key_pcs:
            result.append((t, note, freq, vel, dur))
        else:
            # Try ±1 semitone, prefer the one that's in key
            for offset in [1, -1]:
                new_midi = midi + offset
                if new_midi % 12 in in_key_pcs:
                    new_note = librosa.midi_to_note(new_midi)
                    new_freq = float(librosa.midi_to_hz(new_midi))
                    result.append((t, new_note, new_freq, vel, dur))
                    break
            else:
                result.append((t, note, freq, vel, dur))

    return result


def merge_repeated(notes, max_gap_ms=40):
    """Merge consecutive notes of the same pitch separated by tiny gaps.

    Aubio sometimes splits one sustained note into multiple short notes.
    """
    if not notes:
        return notes

    sorted_notes = sorted(notes, key=lambda x: (x[1], x[0]))

    # Group by note name
    by_note = {}
    for n in sorted_notes:
        by_note.setdefault(n[1], []).append(n)

    merged = []
    for note_name, group in by_note.items():
        group.sort(key=lambda x: x[0])
        current = list(group[0])

        for i in range(1, len(group)):
            t, note, freq, vel, dur = group[i]
            prev_end = current[0] + current[4]
            gap = t - prev_end

            if gap * 1000 <= max_gap_ms:
                # Extend current note
                current[4] = (t + dur) - current[0]
                current[3] = max(current[3], vel)
            else:
                merged.append(tuple(current))
                current = [t, note, freq, vel, dur]

        merged.append(tuple(current))

    merged.sort(key=lambda x: x[0])
    return merged


def apply_all(notes, bpm, key, quantize=True, filter_short=True,
              snap_key=True, merge=True):
    """Apply all post-processing steps in sequence."""
    if filter_short:
        notes = filter_outliers(notes)
    if merge:
        notes = merge_repeated(notes)
    if snap_key:
        notes = snap_to_key(notes, key)
    if quantize:
        notes = quantize_to_grid(notes, bpm)
    return notes
