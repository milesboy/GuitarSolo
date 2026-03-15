"""Map MIDI note numbers to guitar string/fret positions.

Standard tuning: E2(40), A2(45), D3(50), G3(55), B3(59), E4(64)

The max_fret parameter constrains positions to open/low position
playing (typical for fingerstyle arrangements).  Notes that can't
be fretted within the range are flagged.
"""
import librosa

from config import GUITAR_STRINGS

# Default max fret — open position playing (covers most fingerstyle)
DEFAULT_MAX_FRET = 7

# Natural harmonic touch positions: maps MIDI note -> (string_idx, touch_fret)
# These are notes that can only be played as harmonics, not fretted
# String indices: 0=low E, 1=A, 2=D, 3=G, 4=B, 5=high E
NATURAL_HARMONICS = {
    # 5th fret harmonics (4th harmonic = 2 octaves up)
    64: (0, 5),   # E2 * 4 = E4 (also open string 5, but harmonic on 6)
    69: (1, 5),   # A2 * 4 = A4
    74: (2, 5),   # D3 * 4 = D5
    79: (3, 5),   # G3 * 4 = G5
    83: (4, 5),   # B3 * 4 = B5
    88: (5, 5),   # E4 * 4 = E6
    # 7th fret harmonics (3rd harmonic = octave + 5th)
    52: (0, 7),   # E2 * 3 = B3
    57: (1, 7),   # A2 * 3 = E4
    62: (2, 7),   # D3 * 3 = A4
    67: (3, 7),   # G3 * 3 = D5
    71: (4, 7),   # B3 * 3 = F#5
    76: (5, 7),   # E4 * 3 = B5
    # 12th fret harmonics (2nd harmonic = octave)
    52: (0, 12),  # E2 * 2 = E3 (also fretted, prefer fretted)
    57: (1, 12),  # A2 * 2 = A3
    62: (2, 12),  # D3 * 2 = D4
    67: (3, 12),  # G3 * 2 = G4
    71: (4, 12),  # B3 * 2 = B4
    76: (5, 12),  # E4 * 2 = E5
}


def note_to_fret_options(midi_note, max_fret=DEFAULT_MAX_FRET):
    """Return all valid (string_index, fret) positions for a MIDI note.

    String indices are 0-5 (low E to high E).
    Returns list sorted by fret number (lower frets preferred).
    """
    options = []
    for string_idx, open_midi in enumerate(GUITAR_STRINGS):
        fret = midi_note - open_midi
        if 0 <= fret <= max_fret:
            options.append((string_idx, fret))
    return sorted(options, key=lambda x: x[1])


def map_single_note(midi_note, prev_fret=None, prev_string=None,
                    max_fret=DEFAULT_MAX_FRET):
    """Choose the best (string, fret) for a single note.

    Prefers lower frets. Falls back to natural harmonic positions
    for notes above max_fret that match known harmonic frequencies.
    """
    options = note_to_fret_options(midi_note, max_fret)
    if not options:
        # Try natural harmonic position
        if midi_note in NATURAL_HARMONICS:
            return NATURAL_HARMONICS[midi_note]
        return None

    if prev_fret is None:
        return options[0]  # lowest fret

    def proximity_score(opt):
        s, f = opt
        fret_dist = abs(f - prev_fret)
        string_dist = abs(s - (prev_string or 0))
        return fret_dist + string_dist * 2

    return min(options, key=proximity_score)


def map_chord(midi_notes, max_fret=DEFAULT_MAX_FRET):
    """Map a set of simultaneous MIDI notes to string/fret positions.

    Uses a greedy approach that minimizes total fret span while
    avoiding string conflicts.
    """
    if not midi_notes:
        return []

    sorted_notes = sorted(midi_notes)
    assignments = []
    used_strings = set()

    for midi_note in sorted_notes:
        options = note_to_fret_options(midi_note, max_fret)
        available = [(s, f) for s, f in options if s not in used_strings]

        if not available:
            continue

        if assignments:
            existing_frets = [f for _, f in assignments]
            cur_min = min(existing_frets)
            cur_max = max(existing_frets)

            def span_score(opt):
                _, f = opt
                new_min = min(cur_min, f)
                new_max = max(cur_max, f)
                return new_max - new_min

            best = min(available, key=span_score)
        else:
            best = available[0]

        assignments.append(best)
        used_strings.add(best[0])

    return list(zip(sorted_notes, assignments))


def map_notes_sequence(notes, max_fret=DEFAULT_MAX_FRET):
    """Map a sequence of notes to string/fret positions.

    Args:
        notes: list of (time, note_name, freq, velocity, duration)
        max_fret: highest fret to use (default 7 for open position)

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
            note = group[0]
            note_clean = note[1].replace("\u266f", "#").replace("\u266d", "b")
            try:
                midi = librosa.note_to_midi(note_clean)
            except Exception:
                result.append(note + (0, 0))
                continue

            pos = map_single_note(midi, prev_fret, prev_string, max_fret)
            if pos:
                string, fret = pos
                prev_string, prev_fret = string, fret
            else:
                string, fret = 0, 0

            result.append(note + (string, fret))
        else:
            midi_notes = []
            note_map = {}
            for note in group:
                note_clean = note[1].replace("\u266f", "#").replace("\u266d", "b")
                try:
                    midi = librosa.note_to_midi(note_clean)
                    midi_notes.append(midi)
                    note_map[midi] = note
                except Exception:
                    result.append(note + (0, 0))

            if midi_notes:
                assignments = map_chord(midi_notes, max_fret)
                for midi, (string, fret) in assignments:
                    note = note_map[midi]
                    result.append(note + (string, fret))
                    prev_string, prev_fret = string, fret

    return result
