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

from articulation.detector import ArticulationType

CHROMA_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# 64th-note grid: 64 slots per measure in 4/4
GRID_PER_BEAT = 16  # 16 grid slots per quarter note
GRID_PER_MEASURE = 64  # 4 beats * 16

# Grid units -> GP Duration.value
GRID_TO_GP = {
    64: 1,   # whole
    32: 2,   # half
    16: 4,   # quarter
    8:  8,   # eighth
    4:  16,  # sixteenth
    2:  32,  # thirty-second
    1:  64,  # sixty-fourth
}

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


def _snap_grid_dur(grid_units):
    """Snap a duration in grid units to the nearest valid GP duration.

    Returns (gp_value, actual_grid_units) — the GP Duration.value and
    how many grid slots it actually consumes.
    """
    for size in [64, 32, 16, 8, 4, 2, 1]:
        if grid_units >= size:
            return GRID_TO_GP[size], size
    return 64, 1  # sixty-fourth note minimum


def _add_rests(voice, grid_units):
    """Fill a gap with rest beats using the largest durations that fit."""
    remaining = grid_units
    while remaining > 0:
        for size in [64, 32, 16, 8, 4, 2, 1]:
            if size <= remaining:
                beat = guitarpro.Beat(
                    voice, status=guitarpro.BeatStatus.rest)
                beat.duration = guitarpro.Duration(value=GRID_TO_GP[size])
                voice.beats.append(beat)
                remaining -= size
                break


def _make_tie_beat(voice, gp_dur_value, source_beat):
    """Create a tied beat that continues notes from a previous beat."""
    beat = guitarpro.Beat(voice, status=guitarpro.BeatStatus.normal)
    beat.duration = guitarpro.Duration(value=gp_dur_value)
    for src_note in source_beat.notes:
        note = guitarpro.Note(beat)
        note.value = src_note.value
        note.string = src_note.string
        note.velocity = src_note.velocity
        note.type = guitarpro.NoteType.tie
        beat.notes.append(note)
    return beat


def _fill_measure_voice(voice, events, carry_in=None):
    """Fill a measure's voice with properly timed beats and rests.

    Args:
        voice: guitarpro.Voice to populate
        events: list of (grid_pos, grid_dur, beat_builder_fn)
        carry_in: (overflow_grid_units, source_beat) from previous measure
            — tied notes that ring across the bar line

    Returns:
        carry_out: (overflow_grid_units, source_beat) or None
            — if the last note rings past this measure's end
    """
    cursor = 0
    events.sort(key=lambda x: x[0])
    carry_out = None

    # Handle tied notes from previous measure — chain multiple tied
    # beats to fill the full overflow duration
    if carry_in is not None:
        overflow, src_beat = carry_in
        first_event_pos = events[0][0] if events else GRID_PER_MEASURE
        tie_remaining = min(overflow, GRID_PER_MEASURE, first_event_pos)
        while tie_remaining > 0:
            for size in [64, 32, 16, 8, 4, 2, 1]:
                if size <= tie_remaining:
                    tie_beat = _make_tie_beat(
                        voice, GRID_TO_GP[size], src_beat)
                    voice.beats.append(tie_beat)
                    cursor += size
                    tie_remaining -= size
                    break

    for grid_pos, grid_dur, build_beat in events:
        grid_pos = max(grid_pos, cursor)
        if grid_pos >= GRID_PER_MEASURE:
            break

        # Insert rests for the gap before this event
        if grid_pos > cursor:
            _add_rests(voice, grid_pos - cursor)
            cursor = grid_pos

        remaining_in_measure = GRID_PER_MEASURE - cursor
        if remaining_in_measure <= 0:
            break

        if grid_dur <= remaining_in_measure:
            # Note fits entirely in this measure
            gp_value, actual_dur = _snap_grid_dur(grid_dur)
            beat = build_beat(voice, gp_value)
            voice.beats.append(beat)
            cursor += actual_dur
        else:
            # Note overflows into next measure — place what fits here,
            # return the overflow for the next measure as a tie
            gp_value, actual_dur = _snap_grid_dur(remaining_in_measure)
            beat = build_beat(voice, gp_value)
            voice.beats.append(beat)
            carry_out = (grid_dur - actual_dur, beat)
            cursor += actual_dur

    # Fill remainder of measure with rests
    if cursor < GRID_PER_MEASURE:
        _add_rests(voice, GRID_PER_MEASURE - cursor)

    return carry_out


def _time_to_grid(time_sec, bpm, measure_start_sec):
    """Convert an absolute time to a grid position within a measure.

    Returns grid position (0-31) clamped to measure boundaries.
    """
    sec_per_grid = 60.0 / bpm / GRID_PER_BEAT
    offset = time_sec - measure_start_sec
    grid_pos = round(offset / sec_per_grid)
    return max(0, min(grid_pos, GRID_PER_MEASURE - 1))


def _dur_to_grid(dur_sec, bpm):
    """Convert a duration in seconds to grid units."""
    sec_per_grid = 60.0 / bpm / GRID_PER_BEAT
    return max(1, round(dur_sec / sec_per_grid))


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

    # Compute note durations: ring until next onset or end of sustain,
    # whichever comes first (notes ring as long as the WAV shows them
    # sustaining, but never past the next note event)
    sorted_onsets = sorted(note_groups.keys())

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

        # Gap to next onset — note rings at most until then
        if idx + 1 < len(sorted_onsets):
            gap_to_next = sorted_onsets[idx + 1] - onset_time
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
