"""Post-processor for Basic Pitch output.

Applies guitar-specific corrections:
1. Harmonic suppression — remove overtones that BP detects as separate notes
2. Octave correction — snap notes to guitar range
3. False positive filtering — velocity, duration, merge duplicates
4. Duration extension — let notes ring to next onset (guitar sustains naturally)
"""
import librosa
import numpy as np

GUITAR_MIDI_LOW = 40   # E2
GUITAR_MIDI_HIGH = 88  # E6

# Harmonic intervals (semitones) — same as our CQT detector
HARM_INTERVALS = {12, 19, 24, 28, 31, 34, 36}


def suppress_harmonics(notes, mag_ratio=2.0):
    """Remove notes that are harmonics of louder simultaneous notes.

    Groups notes by onset time (within 50ms). Within each group,
    identifies the bass note and removes higher notes that fall on
    harmonic intervals IF they're quieter than the fundamental.

    This is the same logic as our CQT harmonic suppression but
    applied to BP output.
    """
    # Group by onset (within 50ms)
    groups = []
    sorted_notes = sorted(notes, key=lambda n: n[0])

    if not sorted_notes:
        return notes, 0

    current_group = [sorted_notes[0]]
    for n in sorted_notes[1:]:
        if n[0] - current_group[0][0] <= 0.05:
            current_group.append(n)
        else:
            groups.append(current_group)
            current_group = [n]
    groups.append(current_group)

    kept = []
    removed = 0

    for group in groups:
        if len(group) <= 1:
            kept.extend(group)
            continue

        # Convert to MIDI for harmonic analysis
        group_midi = []
        for n in group:
            clean = n[1].replace("\u266f", "#").replace("\u266d", "b")
            try:
                midi = librosa.note_to_midi(clean)
            except Exception:
                midi = 0
            group_midi.append((n, midi))

        # Sort by pitch (low to high)
        group_midi.sort(key=lambda x: x[1])

        # For each note, check if it's a harmonic of a lower, louder note
        suppressed = set()
        for i, (note_i, midi_i) in enumerate(group_midi):
            if i in suppressed:
                continue
            # This note survives — suppress its harmonics
            for j in range(i + 1, len(group_midi)):
                if j in suppressed:
                    continue
                note_j, midi_j = group_midi[j]
                interval = midi_j - midi_i
                if interval in HARM_INTERVALS:
                    # It's a harmonic — suppress if quieter
                    if note_j[3] < note_i[3] * mag_ratio:
                        suppressed.add(j)

        for i, (note, midi) in enumerate(group_midi):
            if i not in suppressed:
                kept.append(note)
            else:
                removed += 1

    return kept, removed


def fix_octave_errors(notes):
    """Snap notes to the nearest guitar-valid octave."""
    corrected = []
    fixes = 0
    for t, name, freq, vel, dur in notes:
        clean = name.replace("\u266f", "#").replace("\u266d", "b")
        try:
            midi = librosa.note_to_midi(clean)
        except Exception:
            corrected.append((t, name, freq, vel, dur))
            continue

        original = midi
        while midi < GUITAR_MIDI_LOW and midi + 12 <= GUITAR_MIDI_HIGH:
            midi += 12
        while midi > GUITAR_MIDI_HIGH and midi - 12 >= GUITAR_MIDI_LOW:
            midi -= 12

        if GUITAR_MIDI_LOW <= midi <= GUITAR_MIDI_HIGH:
            if midi != original:
                fixes += 1
                name = librosa.midi_to_note(midi)
                freq = float(librosa.midi_to_hz(midi))
            corrected.append((t, name, freq, vel, dur))
    return corrected, fixes


def filter_and_merge(notes, min_velocity_pct=0.20, merge_window=0.03):
    """Remove low-velocity notes and merge near-duplicates.

    Args:
        min_velocity_pct: remove notes below this % of max velocity in song
        merge_window: merge same-pitch notes within this time (seconds)
    """
    if not notes:
        return notes, 0

    max_vel = max(n[3] for n in notes)
    threshold = max_vel * min_velocity_pct
    removed = 0

    # Filter by velocity
    filtered = []
    for n in notes:
        if n[3] >= threshold:
            filtered.append(n)
        else:
            removed += 1

    # Merge near-duplicates
    filtered.sort(key=lambda n: (n[1], n[0]))
    merged = []
    i = 0
    while i < len(filtered):
        t, name, freq, vel, dur = filtered[i]
        j = i + 1
        while j < len(filtered):
            t2, name2, _, vel2, dur2 = filtered[j]
            if name2 != name:
                break
            if t2 - t <= merge_window:
                dur = max(dur, (t2 - t) + dur2)
                vel = max(vel, vel2)
                removed += 1
                j += 1
            else:
                break
        merged.append((t, name, freq, vel, dur))
        i = j

    merged.sort(key=lambda n: n[0])
    return merged, removed


def extend_durations(notes):
    """Extend note durations to ring until the next onset in the same range.

    Guitar strings ring naturally until the next pluck on the same string.
    BP durations are often too short. Extend each note to fill the gap
    to the next onset, respecting bass/melody independence.

    Uses the same bass/melody range split as the GP writer.
    """
    BASS_MAX_MIDI = 55  # G3

    def note_range(name):
        clean = name.replace("\u266f", "#").replace("\u266d", "b")
        try:
            midi = librosa.note_to_midi(clean)
            return "bass" if midi <= BASS_MAX_MIDI else "melody"
        except Exception:
            return "melody"

    if not notes:
        return notes, 0

    sorted_notes = sorted(notes, key=lambda n: n[0])
    extended = []
    fixes = 0

    for i, (t, name, freq, vel, dur) in enumerate(sorted_notes):
        rng = note_range(name)

        # Find next onset in same range
        next_onset = None
        for j in range(i + 1, len(sorted_notes)):
            future_rng = note_range(sorted_notes[j][1])
            if future_rng == rng or (rng == "bass" and future_rng == "bass") or \
               (rng == "melody" and future_rng == "melody"):
                next_onset = sorted_notes[j][0]
                break

        if next_onset is not None:
            gap = next_onset - t
            # Extend to fill gap, but cap at 4 seconds
            new_dur = min(gap, 4.0)
            # Only extend, never shorten
            if new_dur > dur:
                fixes += 1
                dur = new_dur

        extended.append((t, name, freq, vel, dur))

    return extended, fixes


def postprocess_bp(notes, y=None, sr=None, verbose=True):
    """Full post-processing pipeline for Basic Pitch output."""
    if verbose:
        print(f"\n=== BP Post-Processing ===")
        print(f"  Input: {len(notes)} notes")

    # Step 1: Harmonic suppression
    notes, harm_removed = suppress_harmonics(notes)
    if verbose:
        print(f"  Harmonics removed: {harm_removed}")

    # Step 2: Octave correction
    notes, octave_fixes = fix_octave_errors(notes)
    if verbose:
        print(f"  Octave fixes: {octave_fixes}")

    # Step 3: Filter and merge
    notes, filter_removed = filter_and_merge(notes)
    if verbose:
        print(f"  Filtered/merged: {filter_removed}")

    # Step 4: Extend durations
    notes, dur_extended = extend_durations(notes)
    if verbose:
        print(f"  Durations extended: {dur_extended}")

    if verbose:
        print(f"  Output: {len(notes)} notes")

    return notes
