"""Write Guitar Pro (.gp5) files with articulation support.

Uses pyguitarpro to create tablature with proper guitar techniques:
hammer-ons, pull-offs, slides, bends.

Track 1: Melody/fingerpicking (detected notes with articulations)
Track 2: Chords (muted by default — reference track)
"""
import guitarpro
from guitarpro.models import BendPoint
import math

from articulation.detector import ArticulationType

CHROMA_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

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
# Aliases
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


def _duration_to_gp(dur_beats):
    """Convert a duration in beats to the nearest Guitar Pro Duration value."""
    if dur_beats >= 3.0:
        return 1   # whole
    elif dur_beats >= 1.5:
        return 2   # half
    elif dur_beats >= 0.75:
        return 4   # quarter
    elif dur_beats >= 0.375:
        return 8   # eighth
    elif dur_beats >= 0.1875:
        return 16  # sixteenth
    else:
        return 32  # thirty-second


def write_guitarpro(fretted_notes, articulations, bpm, key="C major",
                    chords=None, title="", artist="",
                    output_path="output.gp5"):
    """Create a Guitar Pro file with melody track and muted chord track.

    Args:
        fretted_notes: list of (time, note_name, freq, vel, dur, string, fret)
        articulations: list of ArticulationType, one per note
        bpm: tempo
        key: key signature string
        chords: list of (time, chord_name) — placed on separate muted track
        title: song title
        artist: artist name
        output_path: where to save the .gp5 file

    Returns:
        Path to the written file.
    """
    song = guitarpro.Song()
    song.title = title or "Untitled"
    song.artist = artist or ""
    song.tempo = int(bpm)

    # --- Track 1: Melody ---
    melody_track = song.tracks[0]
    melody_track.name = "Melody"
    melody_track.number = 1
    melody_track.channel.channel = 0
    melody_track.channel.instrument = 25  # Acoustic Guitar (steel)
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

    beats_per_sec = bpm / 60.0
    beats_per_measure = 4
    sec_per_measure = beats_per_measure / beats_per_sec

    # Determine number of measures needed
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

    # Ensure enough measures for both tracks
    while len(song.measureHeaders) < num_measures:
        header = guitarpro.MeasureHeader()
        song.measureHeaders.append(header)
        melody_track.measures.append(guitarpro.Measure(melody_track, header))

    # Add chord track with matching measures
    for header in song.measureHeaders:
        chord_track.measures.append(guitarpro.Measure(chord_track, header))
    song.tracks.append(chord_track)

    # ==========================================
    # Populate Track 1: Melody notes
    # ==========================================
    note_groups = {}
    for i, n in enumerate(fretted_notes):
        t = round(n[0], 3)
        if t not in note_groups:
            note_groups[t] = []
        art = articulations[i] if i < len(articulations) else ArticulationType.PICKED
        note_groups[t].append((n, art))

    for onset_time in sorted(note_groups.keys()):
        group = note_groups[onset_time]

        measure_idx = int(onset_time / sec_per_measure)
        if measure_idx >= len(melody_track.measures):
            continue

        measure = melody_track.measures[measure_idx]
        beat = guitarpro.Beat(measure.voices[0])
        dur_beats = group[0][0][4] * beats_per_sec
        beat.duration = guitarpro.Duration(value=_duration_to_gp(dur_beats))

        for (note_data, art) in group:
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
                bend = guitarpro.BendEffect()
                bend.type = guitarpro.BendType.bend
                bend.value = 100
                bend.points = [
                    BendPoint(0, 0),
                    BendPoint(6, 100),
                    BendPoint(12, 100),
                ]
                note.effect.bend = bend

            beat.notes.append(note)

        measure.voices[0].beats.append(beat)

    # ==========================================
    # Populate Track 2: Chords
    # ==========================================
    if chords:
        for i, (chord_time, chord_name) in enumerate(chords):
            voicing = CHORD_VOICINGS.get(chord_name)
            if not voicing:
                continue

            # Chord duration: until next chord or 2 seconds
            if i + 1 < len(chords):
                chord_dur = chords[i + 1][0] - chord_time
            else:
                chord_dur = 2.0
            chord_dur = max(chord_dur, 0.25)

            measure_idx = int(chord_time / sec_per_measure)
            if measure_idx >= len(chord_track.measures):
                continue

            measure = chord_track.measures[measure_idx]
            beat = guitarpro.Beat(measure.voices[0])
            dur_beats = chord_dur * beats_per_sec
            beat.duration = guitarpro.Duration(value=_duration_to_gp(dur_beats))

            for (gp_string, fret) in voicing:
                note = guitarpro.Note(beat)
                note.value = fret
                note.string = gp_string
                note.velocity = 80
                beat.notes.append(note)

            measure.voices[0].beats.append(beat)

    guitarpro.write(song, output_path)
    return output_path
