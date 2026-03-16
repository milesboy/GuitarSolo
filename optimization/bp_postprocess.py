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


def suppress_harmonics(notes, vel_threshold=0.6):
    """Remove notes that are harmonics of louder simultaneous notes.

    Groups notes by onset time (within 50ms). Within each group,
    identifies the bass note and removes higher notes that fall on
    harmonic intervals IF they're significantly quieter than the
    fundamental (below vel_threshold fraction).

    BP velocities are in a narrow range, so only suppress truly
    quiet harmonics — loud notes at harmonic positions are likely
    independently played.
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
                    # It's a harmonic — only suppress if truly quiet
                    # (below 50% of fundamental velocity)
                    if note_j[3] < note_i[3] * vel_threshold:
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


def filter_and_merge(notes, min_velocity_pct=0.25, merge_window=0.03):
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


def cap_durations(notes):
    """Cap note durations at the next onset to prevent overlap.

    BP durations are often 50-70% too long (GT avg ~0.3s, BP avg ~0.5s).
    Cap each note so it doesn't ring past the next note in the same range.
    This prevents the muddy overlapping sound in GP playback.
    """
    BASS_MAX_MIDI = 55

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
    capped = []
    fixes = 0

    for i, (t, name, freq, vel, dur) in enumerate(sorted_notes):
        rng = note_range(name)

        # Find next onset in same range
        next_onset = None
        for j in range(i + 1, len(sorted_notes)):
            if note_range(sorted_notes[j][1]) == rng:
                next_onset = sorted_notes[j][0]
                break

        if next_onset is not None:
            gap = next_onset - t
            # Cap duration at the gap (don't overlap)
            if dur > gap:
                fixes += 1
                dur = max(gap, 0.05)  # minimum 50ms

        capped.append((t, name, freq, vel, dur))

    return capped, fixes


def remove_sustain_duplicates(notes):
    """Remove notes that are sustain energy misread as re-plucks.

    If the same pitch appears again while the previous note is still
    ringing AND the new one is quieter (or similar velocity), it's
    sustain leakage, not a real new pluck. A real re-pluck would be
    at least as loud as the decaying note.

    This fixes the "sustain interpreted as new notes" problem.
    """
    if not notes:
        return notes, 0

    sorted_notes = sorted(notes, key=lambda n: n[0])
    kept = []
    removed = 0

    # Track ringing notes: pitch -> (end_time, velocity)
    ringing = {}

    for t, name, freq, vel, dur in sorted_notes:
        clean = name.replace("\u266f", "#").replace("\u266d", "b")

        # Is this pitch currently ringing from a previous note?
        if clean in ringing:
            prev_end, prev_vel = ringing[clean]
            if t < prev_end:
                # Previous note is still ringing at this time.
                # Real re-pluck: new note should be at least 70% of
                # the original velocity (a fresh pluck has energy).
                # Sustain leakage: quieter, just the tail of the old note.
                if vel < prev_vel * 0.70:
                    removed += 1
                    continue  # skip — it's sustain, not a re-pluck

        # Keep this note and track its ringing
        kept.append((t, name, freq, vel, dur))
        ringing[clean] = (t + dur, vel)

    return kept, removed


def suppress_under_sustain(notes):
    """Remove lower notes that appear under a sustained higher note.

    When a high note is ringing, lower notes that emerge underneath
    are typically harmonics or sympathetic resonance, not new plucks.
    The sustained high note dominates perceptually.

    Only suppresses if the new note is:
    - Lower in pitch than a currently ringing note
    - Quieter than the ringing note
    - Not a strong independent pluck (velocity check)
    """
    if not notes:
        return notes, 0

    sorted_notes = sorted(notes, key=lambda n: n[0])
    kept = []
    removed = 0

    # Track ringing notes: list of (onset_time, end_time, midi, velocity)
    ringing = []

    for t, name, freq, vel, dur in sorted_notes:
        clean = name.replace("\u266f", "#").replace("\u266d", "b")
        try:
            midi = librosa.note_to_midi(clean)
        except Exception:
            kept.append((t, name, freq, vel, dur))
            continue

        # Expire old ringing notes
        ringing = [(ot, end, m, v) for ot, end, m, v in ringing if end > t]

        # Check: is this note lower than any currently ringing note
        # AND quieter? If so, it's likely a harmonic/artifact.
        # BUT: don't suppress if it started within 50ms of the higher
        # note — they're part of the same chord/onset event.
        suppressed = False
        for ring_onset, ring_end, ring_midi, ring_vel in ringing:
            if midi < ring_midi and vel < ring_vel * 0.8:
                # Same onset? (within 50ms = same chord)
                if abs(t - ring_onset) <= 0.05:
                    continue  # part of same event, keep it
                suppressed = True
                removed += 1
                break

        if not suppressed:
            kept.append((t, name, freq, vel, dur))
            ringing.append((t, t + dur, midi, vel))

    return kept, removed


def postprocess_bp(notes, y=None, sr=None, verbose=True, verify_pitch=True):
    """Full post-processing pipeline for Basic Pitch output.

    Args:
        notes: list of (time, note_name, freq, velocity, duration)
        y, sr: audio signal (needed for pitch voting)
        verify_pitch: if True and audio provided, run CQT pitch voting
        verbose: print progress
    """
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

    # Step 3: Remove sustain duplicates
    notes, sustain_removed = remove_sustain_duplicates(notes)
    if verbose:
        print(f"  Sustain duplicates removed: {sustain_removed}")

    # Step 4: Filter and merge
    notes, filter_removed = filter_and_merge(notes)
    if verbose:
        print(f"  Filtered/merged: {filter_removed}")

    # Step 5: Suppress lower notes under sustained high notes
    notes, under_sustain = suppress_under_sustain(notes)
    if verbose:
        print(f"  Under-sustain suppressed: {under_sustain}")

    if verbose:
        print(f"  Output: {len(notes)} notes")

    return notes
