"""Map MIDI note numbers to guitar string/fret positions.

Standard tuning: E2(40), A2(45), D3(50), G3(55), B3(59), E4(64)
Each string covers frets 0-24.
"""
import librosa

from config import GUITAR_STRINGS, NUM_FRETS


def note_to_fret_options(midi_note):
    """Return all valid (string_index, fret) positions for a MIDI note.

    String indices are 0-5 (low E to high E).
    Returns list sorted by fret number (lower frets preferred).
    """
    options = []
    for string_idx, open_midi in enumerate(GUITAR_STRINGS):
        fret = midi_note - open_midi
        if 0 <= fret <= NUM_FRETS:
            options.append((string_idx, fret))
    return sorted(options, key=lambda x: x[1])


def map_single_note(midi_note, prev_fret=None, prev_string=None):
    """Choose the best (string, fret) for a single note.

    Prefers lower frets, but considers proximity to previous note
    for minimal hand movement.
    """
    options = note_to_fret_options(midi_note)
    if not options:
        return None

    if prev_fret is None:
        return options[0]  # lowest fret

    # Score by proximity to previous position
    def proximity_score(opt):
        s, f = opt
        fret_dist = abs(f - prev_fret)
        string_dist = abs(s - (prev_string or 0))
        return fret_dist + string_dist * 2  # penalize string jumps

    return min(options, key=proximity_score)


def map_chord(midi_notes):
    """Map a set of simultaneous MIDI notes to string/fret positions.

    Uses a greedy approach that minimizes total fret span while
    avoiding string conflicts.
    """
    if not midi_notes:
        return []

    # Sort notes low to high
    sorted_notes = sorted(midi_notes)
    assignments = []
    used_strings = set()

    for midi_note in sorted_notes:
        options = note_to_fret_options(midi_note)
        # Filter out already-used strings
        available = [(s, f) for s, f in options if s not in used_strings]

        if not available:
            continue  # can't place this note

        # If we have existing assignments, minimize fret span
        if assignments:
            existing_frets = [f for _, f in assignments]
            min_fret = min(existing_frets)
            max_fret = max(existing_frets)

            def span_score(opt):
                _, f = opt
                new_min = min(min_fret, f)
                new_max = max(max_fret, f)
                return new_max - new_min

            best = min(available, key=span_score)
        else:
            best = available[0]  # lowest fret

        assignments.append(best)
        used_strings.add(best[0])

    return list(zip(sorted_notes, assignments))


def map_notes_sequence(notes):
    """Map a sequence of notes to string/fret positions.

    Args:
        notes: list of (time, note_name, freq, velocity, duration)

    Returns:
        list of (time, note_name, freq, velocity, duration, string, fret)
    """
    result = []
    prev_fret = None
    prev_string = None

    # Group notes by time (simultaneous = chord)
    groups = {}
    for n in notes:
        t = round(n[0], 3)
        groups.setdefault(t, []).append(n)

    for t in sorted(groups.keys()):
        group = groups[t]

        if len(group) == 1:
            # Single note
            note = group[0]
            note_clean = note[1].replace("♯", "#").replace("♭", "b")
            try:
                midi = librosa.note_to_midi(note_clean)
            except Exception:
                result.append(note + (0, 0))
                continue

            pos = map_single_note(midi, prev_fret, prev_string)
            if pos:
                string, fret = pos
                prev_string, prev_fret = string, fret
            else:
                string, fret = 0, 0

            result.append(note + (string, fret))
        else:
            # Chord — map simultaneously
            midi_notes = []
            note_map = {}
            for note in group:
                note_clean = note[1].replace("♯", "#").replace("♭", "b")
                try:
                    midi = librosa.note_to_midi(note_clean)
                    midi_notes.append(midi)
                    note_map[midi] = note
                except Exception:
                    result.append(note + (0, 0))

            if midi_notes:
                assignments = map_chord(midi_notes)
                for midi, (string, fret) in assignments:
                    note = note_map[midi]
                    result.append(note + (string, fret))
                    prev_string, prev_fret = string, fret

    return result
