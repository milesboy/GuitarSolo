"""Write Guitar Pro (.gp5) files with articulation support.

Uses pyguitarpro to create tablature with proper guitar techniques:
hammer-ons, pull-offs, slides, bends.

Track 1: Melody/fingerpicking (detected notes with articulations)
Track 2: Chords (muted by default — reference track)

Timing: all note onsets are quantized to a 32nd-note grid and rests
are inserted for gaps, so GP playback matches the original audio timing.
"""
import guitarpro
from guitarpro.models import BendPoint
import math

import librosa

from articulation.detector import ArticulationType

# Voice ranges for independent sustain — bass notes ring through
# melody events and vice versa, just like real fingerstyle guitar
# where thumb (bass) and fingers (melody) operate independently.
BASS_RANGE = (40, 55)    # E2 to G3 — thumb/bass strings
MELODY_RANGE = (56, 88)  # G#3 to E6 — finger/melody strings + harmonics
MAX_GUITAR_MIDI = 88     # E6 — highest harmonic (5th fret, 1st string)


def _note_range(note_name):
    """Classify a note into bass or melody range."""
    try:
        midi = librosa.note_to_midi(
            note_name.replace("\u266f", "#").replace("\u266d", "b"))
    except Exception:
        return "melody"
    if midi <= BASS_RANGE[1]:
        return "bass"
    return "melody"


def _group_range(group):
    """Determine the range(s) present in a note group."""
    ranges = set()
    for (note_data, _art) in group:
        ranges.add(_note_range(note_data[1]))
    return ranges

CHROMA_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Internal grid: 64th notes for precise detection
GRID_PER_BEAT = 16  # 16 grid slots per quarter note
GRID_PER_MEASURE = 64  # 4 beats * 16

# Output quantization: snap to 16th notes (4 grid units)
# Removes human timing imprecision — most guitar music uses 8th/16th
# note subdivisions at fastest.  Internal 64th grid ensures we don't
# miss anything; quantization rounds to the musical grid.
QUANTIZE = 4  # grid units per 16th note

# Duration table: (grid_units, gp_value, isDotted)
# Includes dotted notes for natural rhythmic values (e.g. dotted quarter
# = quarter + eighth = 1.5 beats). Descending by grid size.
DURATION_TABLE = [
    (64, 1,  False),  # whole
    (48, 2,  True),   # dotted half
    (32, 2,  False),  # half
    (24, 4,  True),   # dotted quarter
    (16, 4,  False),  # quarter
    (12, 8,  True),   # dotted eighth
    (8,  8,  False),  # eighth
    (6,  16, True),   # dotted sixteenth
    (4,  16, False),  # sixteenth
]

# Chord voicings as (string, fret) lists — open position shapes
# String numbering: 1=high E, 2=B, 3=G, 4=D, 5=A, 6=low E
CHORD_VOICINGS = {
    "C":  [(1,0),(2,1),(3,0),(4,2),(5,3)],
    "D":  [(1,2),(2,3),(3,2),(4,0)],
    "Dm": [(1,1),(2,3),(3,2),(4,0)],
    "E":  [(1,0),(2,0),(3,1),(4,2),(5,2),(6,0)],
    "Em": [(1,0),(2,0),(3,0),(4,2),(5,2),(6,0)],
    "F":  [(1,1),(2,1),(3,2),(4,3),(5,3),(6,1)],
    "Fm": [(1,1),(2,1),(3,1),(4,3),(5,3),(6,1)],
    "G":  [(1,3),(2,0),(3,0),(4,0),(5,2),(6,3)],
    "A":  [(1,0),(2,2),(3,2),(4,2),(5,0)],
    "Am": [(1,0),(2,1),(3,2),(4,2),(5,0)],
    "B":  [(1,2),(2,4),(3,4),(4,4),(5,2)],
    "Bm": [(1,2),(2,3),(3,4),(4,4),(5,2)],
    "C#": [(1,1),(2,2),(3,3),(4,3),(5,4)],
    "C#m":[(1,0),(2,2),(3,2),(4,1),(5,4)],
    "D#": [(1,3),(2,4),(3,3),(4,1),(5,6)],
    "D#m":[(1,2),(2,4),(3,3),(4,1),(5,6)],
    "F#": [(1,2),(2,2),(3,3),(4,4),(5,4),(6,2)],
    "F#m":[(1,2),(2,2),(3,2),(4,4),(5,4),(6,2)],
    "G#": [(1,4),(2,4),(3,5),(4,6),(5,6),(6,4)],
    "G#m":[(1,4),(2,4),(3,4),(4,6),(5,6),(6,4)],
    "A#": [(1,1),(2,1),(3,3),(4,3),(5,1)],
    "A#m":[(1,1),(2,1),(3,2),(4,3),(5,1)],
}
CHORD_VOICINGS["Db"] = CHORD_VOICINGS["C#"]
CHORD_VOICINGS["Dbm"] = CHORD_VOICINGS["C#m"]
CHORD_VOICINGS["Eb"] = CHORD_VOICINGS["D#"]
CHORD_VOICINGS["Ebm"] = CHORD_VOICINGS["D#m"]
CHORD_VOICINGS["Gb"] = CHORD_VOICINGS["F#"]
CHORD_VOICINGS["Gbm"] = CHORD_VOICINGS["F#m"]
CHORD_VOICINGS["Ab"] = CHORD_VOICINGS["G#"]
CHORD_VOICINGS["Abm"] = CHORD_VOICINGS["G#m"]
CHORD_VOICINGS["Bb"] = CHORD_VOICINGS["A#"]
CHORD_VOICINGS["Bbm"] = CHORD_VOICINGS["A#m"]


def _quantize_pos(grid_pos):
    """Snap a grid position to the nearest quantized boundary."""
    return round(grid_pos / QUANTIZE) * QUANTIZE


def _quantize_dur(grid_units):
    """Snap a duration DOWN to the nearest quantized value.

    Floors instead of rounding so notes never bleed past the next onset.
    """
    return max(QUANTIZE, (grid_units // QUANTIZE) * QUANTIZE)


def _make_duration(grid_units):
    """Create a guitarpro.Duration for the given grid units.

    Finds the best match from DURATION_TABLE (largest that fits),
    including dotted notes.

    Returns (guitarpro.Duration, actual_grid_units).
    """
    for size, gp_value, dotted in DURATION_TABLE:
        if size <= grid_units:
            return guitarpro.Duration(value=gp_value, isDotted=dotted), size
    # Minimum: sixteenth
    return guitarpro.Duration(value=16), QUANTIZE


def _add_rests(voice, grid_units):
    """Fill a gap with rest beats using the largest durations that fit."""
    remaining = grid_units
    while remaining >= QUANTIZE:
        dur, size = _make_duration(remaining)
        beat = guitarpro.Beat(voice, status=guitarpro.BeatStatus.rest)
        beat.duration = dur
        voice.beats.append(beat)
        remaining -= size


def _add_ties(voice, grid_units, source_beat):
    """Fill a duration with tied beats using the largest values that fit."""
    remaining = grid_units
    while remaining >= QUANTIZE:
        dur, size = _make_duration(remaining)
        beat = guitarpro.Beat(voice, status=guitarpro.BeatStatus.normal)
        beat.duration = dur
        for src_note in source_beat.notes:
            note = guitarpro.Note(beat)
            note.value = src_note.value
            note.string = src_note.string
            note.velocity = src_note.velocity
            note.type = guitarpro.NoteType.tie
            beat.notes.append(note)
        voice.beats.append(beat)
        remaining -= size


def _fill_measure_voice(voice, events, carry_in=None):
    """Fill a measure's voice with properly timed beats and rests.

    Notes never overlap: each note's duration is capped at the gap to
    the next note onset. When a new note plays, the old one stops.

    Args:
        voice: guitarpro.Voice to populate
        events: list of (grid_pos, grid_dur, beat_builder_fn)
        carry_in: (overflow_grid_units, source_beat) from previous measure

    Returns:
        carry_out: (overflow_grid_units, source_beat) or None
    """
    cursor = 0
    events.sort(key=lambda x: x[0])
    carry_out = None

    # Handle tied notes from previous measure
    if carry_in is not None:
        overflow, src_beat = carry_in
        first_event_pos = events[0][0] if events else GRID_PER_MEASURE
        tie_dur = min(overflow, GRID_PER_MEASURE, first_event_pos)
        tie_dur = _quantize_dur(tie_dur) if tie_dur >= QUANTIZE else 0
        tie_dur = min(tie_dur, first_event_pos)
        if tie_dur >= QUANTIZE:
            _add_ties(voice, tie_dur, src_beat)
            cursor = tie_dur

    for i, (grid_pos, grid_dur, build_beat) in enumerate(events):
        grid_pos = max(grid_pos, cursor)
        if grid_pos >= GRID_PER_MEASURE:
            break

        # Insert rests for the gap before this event
        if grid_pos > cursor:
            _add_rests(voice, grid_pos - cursor)
            cursor = grid_pos

        # Cap duration: note stops when the next note starts (no overlap)
        if i + 1 < len(events):
            next_pos = max(events[i + 1][0], grid_pos + QUANTIZE)
            grid_dur = min(grid_dur, next_pos - grid_pos)

        remaining_in_measure = GRID_PER_MEASURE - cursor
        if remaining_in_measure < QUANTIZE:
            break

        if grid_dur <= remaining_in_measure:
            dur, actual_dur = _make_duration(grid_dur)
            beat = build_beat(voice, dur.value)
            beat.duration = dur  # override with possibly dotted duration
            voice.beats.append(beat)
            cursor += actual_dur
        else:
            # Note overflows into next measure
            dur, actual_dur = _make_duration(remaining_in_measure)
            beat = build_beat(voice, dur.value)
            beat.duration = dur
            voice.beats.append(beat)
            carry_out = (grid_dur - actual_dur, beat)
            cursor += actual_dur

    # Fill remainder of measure with rests
    if cursor < GRID_PER_MEASURE:
        _add_rests(voice, GRID_PER_MEASURE - cursor)

    return carry_out


def _time_to_grid(time_sec, bpm, measure_start_sec):
    """Convert an absolute time to a quantized grid position within a measure.

    Snaps to nearest 16th-note boundary (QUANTIZE grid units).
    """
    sec_per_grid = 60.0 / bpm / GRID_PER_BEAT
    offset = time_sec - measure_start_sec
    grid_pos = round(offset / sec_per_grid)
    # Snap to quantized boundary
    grid_pos = _quantize_pos(grid_pos)
    return max(0, min(grid_pos, GRID_PER_MEASURE - QUANTIZE))


def _dur_to_grid(dur_sec, bpm):
    """Convert a duration in seconds to quantized grid units.

    Snaps to nearest 16th-note duration (minimum QUANTIZE units).
    """
    sec_per_grid = 60.0 / bpm / GRID_PER_BEAT
    raw = round(dur_sec / sec_per_grid)
    return _quantize_dur(raw)


def _make_note_beat_builder(note_data_list):
    """Create a beat builder function for a group of simultaneous notes.

    Args:
        note_data_list: list of ((note_tuple, articulation), ...)
    """
    def build(voice, gp_dur_value):
        beat = guitarpro.Beat(voice, status=guitarpro.BeatStatus.normal)
        beat.duration = guitarpro.Duration(value=gp_dur_value)

        for (note_data, art) in note_data_list:
            _, _, _, vel, _, string_idx, fret = note_data
            gp_string = 6 - string_idx

            note = guitarpro.Note(beat)
            note.value = fret
            note.string = gp_string
            note.velocity = min(max(vel, 1), 127)
            note.type = guitarpro.NoteType.normal
            note.effect = guitarpro.NoteEffect()

            if art == ArticulationType.HAMMER_ON:
                note.effect.hammer = True
            elif art == ArticulationType.PULL_OFF:
                note.effect.hammer = True
            elif art in (ArticulationType.SLIDE_UP, ArticulationType.SLIDE_DOWN):
                note.effect.slides = [guitarpro.SlideType.shiftSlideTo]
            elif art == ArticulationType.BEND:
                bend_effect = guitarpro.BendEffect()
                bend_effect.type = guitarpro.BendType.bend
                bend_effect.value = 100
                bend_effect.points = [
                    BendPoint(0, 0),
                    BendPoint(6, 100),
                    BendPoint(12, 100),
                ]
                note.effect.bend = bend_effect

            beat.notes.append(note)
        return beat
    return build


def _make_chord_beat_builder(voicing):
    """Create a beat builder function for a chord."""
    def build(voice, gp_dur_value):
        beat = guitarpro.Beat(voice, status=guitarpro.BeatStatus.normal)
        beat.duration = guitarpro.Duration(value=gp_dur_value)
        for (gp_string, fret) in voicing:
            note = guitarpro.Note(beat)
            note.value = fret
            note.string = gp_string
            note.velocity = 80
            note.type = guitarpro.NoteType.normal
            beat.notes.append(note)
        return beat
    return build


def write_guitarpro(fretted_notes, articulations, bpm, key="C major",
                    chords=None, title="", artist="",
                    output_path="output.gp5"):
    """Create a Guitar Pro file with melody track and muted chord track.

    All note onset times are quantized to a 32nd-note grid and gaps are
    filled with rests so playback timing matches the original audio.
    """
    song = guitarpro.Song()
    song.title = title or "Untitled"
    song.artist = artist or ""
    song.tempo = int(bpm)

    beats_per_sec = bpm / 60.0
    sec_per_measure = 4.0 / beats_per_sec  # 4/4 time

    # --- Track 1: Melody ---
    melody_track = song.tracks[0]
    melody_track.name = "Melody"
    melody_track.number = 1
    melody_track.port = 0
    melody_track.offset = -12  # guitar sounds one octave lower than written
    melody_track.channel.channel = 0
    melody_track.channel.instrument = 25
    melody_track.isPercussionTrack = False
    melody_track.strings = [
        guitarpro.GuitarString(1, 64),
        guitarpro.GuitarString(2, 59),
        guitarpro.GuitarString(3, 55),
        guitarpro.GuitarString(4, 50),
        guitarpro.GuitarString(5, 45),
        guitarpro.GuitarString(6, 40),
    ]

    # --- Track 2: Chords (muted) ---
    chord_track = guitarpro.Track(song)
    chord_track.name = "Chords"
    chord_track.number = 2
    chord_track.port = 0
    chord_track.offset = -12  # guitar octave transposition
    chord_track.isMute = True
    chord_track.channel.channel = 2
    chord_track.channel.instrument = 25
    chord_track.channel.volume = 80
    chord_track.isPercussionTrack = False
    chord_track.strings = [
        guitarpro.GuitarString(1, 64),
        guitarpro.GuitarString(2, 59),
        guitarpro.GuitarString(3, 55),
        guitarpro.GuitarString(4, 50),
        guitarpro.GuitarString(5, 45),
        guitarpro.GuitarString(6, 40),
    ]

    # Determine total measures needed
    max_time = 0.0
    if fretted_notes:
        max_time = max(max_time, max(n[0] + n[4] for n in fretted_notes))
    if chords:
        max_time = max(max_time, chords[-1][0] + 2.0)

    if max_time == 0.0:
        song.tracks.append(chord_track)
        guitarpro.write(song, output_path)
        return output_path

    num_measures = max(1, math.ceil(max_time / sec_per_measure) + 1)

    # Create measure headers and measures for both tracks
    while len(song.measureHeaders) < num_measures:
        header = guitarpro.MeasureHeader()
        song.measureHeaders.append(header)
        melody_track.measures.append(guitarpro.Measure(melody_track, header))

    for header in song.measureHeaders:
        chord_track.measures.append(guitarpro.Measure(chord_track, header))
    song.tracks.append(chord_track)

    # ==========================================
    # Build per-measure event lists for melody
    # ==========================================
    # Group simultaneous notes
    note_groups = {}
    for i, n in enumerate(fretted_notes):
        t = round(n[0], 3)
        if t not in note_groups:
            note_groups[t] = []
        art = articulations[i] if i < len(articulations) else ArticulationType.PICKED
        note_groups[t].append((n, art))

    # Compute note durations with range-aware sustain:
    # Bass notes (E2-G3) ring independently of melody notes (G#3-D6).
    # A bass note only stops when the NEXT BASS event starts, not when
    # a melody event occurs (and vice versa). This models fingerstyle
    # guitar where thumb and fingers operate independently.
    sorted_onsets = sorted(note_groups.keys())

    # Pre-compute ranges for each onset
    onset_ranges = {}
    for onset_time in sorted_onsets:
        onset_ranges[onset_time] = _group_range(note_groups[onset_time])

    melody_events_by_measure = {m: [] for m in range(num_measures)}

    for idx, onset_time in enumerate(sorted_onsets):
        group = note_groups[onset_time]
        measure_idx = int(onset_time / sec_per_measure)
        if measure_idx >= num_measures:
            continue

        measure_start = measure_idx * sec_per_measure
        grid_pos = _time_to_grid(onset_time, bpm, measure_start)

        # Detected sustain duration from WAV analysis
        sustain_dur = max(n[0][4] for n in group)

        # Find the next onset that shares a voice range with this one.
        # Bass notes ring past melody onsets; melody rings past bass.
        my_ranges = onset_ranges[onset_time]
        gap_to_next = None
        for future_idx in range(idx + 1, len(sorted_onsets)):
            future_time = sorted_onsets[future_idx]
            future_ranges = onset_ranges[future_time]
            if my_ranges & future_ranges:  # overlapping ranges
                gap_to_next = future_time - onset_time
                break

        if gap_to_next is not None:
            ring_dur = min(sustain_dur, gap_to_next)
        else:
            ring_dur = sustain_dur

        grid_dur = _dur_to_grid(ring_dur, bpm)

        builder = _make_note_beat_builder(group)
        melody_events_by_measure[measure_idx].append(
            (grid_pos, grid_dur, builder))

    # Fill each measure with properly timed beats + rests,
    # carrying tied notes across bar lines
    carry = None
    for m_idx in range(num_measures):
        if m_idx >= len(melody_track.measures):
            break
        measure = melody_track.measures[m_idx]
        voice = measure.voices[0]
        voice.beats.clear()
        events = melody_events_by_measure[m_idx]
        carry = _fill_measure_voice(voice, events, carry_in=carry)

    # ==========================================
    # Build per-measure event lists for chords
    # ==========================================
    if chords:
        chord_events_by_measure = {m: [] for m in range(num_measures)}

        for i, (chord_time, chord_name) in enumerate(chords):
            voicing = CHORD_VOICINGS.get(chord_name)
            if not voicing:
                continue

            if i + 1 < len(chords):
                chord_dur = chords[i + 1][0] - chord_time
            else:
                chord_dur = 2.0
            chord_dur = max(chord_dur, 0.25)

            measure_idx = int(chord_time / sec_per_measure)
            if measure_idx >= num_measures:
                continue

            measure_start = measure_idx * sec_per_measure
            grid_pos = _time_to_grid(chord_time, bpm, measure_start)
            grid_dur = _dur_to_grid(chord_dur, bpm)

            builder = _make_chord_beat_builder(voicing)
            chord_events_by_measure[measure_idx].append(
                (grid_pos, grid_dur, builder))

        chord_carry = None
        for m_idx in range(num_measures):
            if m_idx >= len(chord_track.measures):
                break
            measure = chord_track.measures[m_idx]
            voice = measure.voices[0]
            voice.beats.clear()
            events = chord_events_by_measure[m_idx]
            chord_carry = _fill_measure_voice(
                voice, events, carry_in=chord_carry)

    guitarpro.write(song, output_path)
    return output_path
