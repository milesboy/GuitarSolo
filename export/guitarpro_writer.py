"""Write Guitar Pro (.gp5) files with articulation support.

Uses pyguitarpro to create tablature with proper guitar techniques:
hammer-ons, pull-offs, slides, bends.
"""
import guitarpro
from guitarpro.models import BendPoint
import math

from articulation.detector import ArticulationType


def _duration_to_gp(dur_beats):
    """Convert a duration in beats to the nearest Guitar Pro Duration value.

    GP durations: whole=1, half=2, quarter=4, eighth=8, sixteenth=16, etc.
    """
    if dur_beats >= 3.0:
        return 1  # whole
    elif dur_beats >= 1.5:
        return 2  # half
    elif dur_beats >= 0.75:
        return 4  # quarter
    elif dur_beats >= 0.375:
        return 8  # eighth
    elif dur_beats >= 0.1875:
        return 16  # sixteenth
    else:
        return 32  # thirty-second


def write_guitarpro(fretted_notes, articulations, bpm, key="C major",
                    chords=None, title="", artist="",
                    output_path="output.gp5"):
    """Create a Guitar Pro file from detected notes with articulations.

    Args:
        fretted_notes: list of (time, note_name, freq, vel, dur, string, fret)
        articulations: list of ArticulationType, one per note
        bpm: tempo
        key: key signature string
        chords: optional chord list
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

    # Set up a guitar track
    track = song.tracks[0]
    track.name = "Guitar"
    track.channel.instrument = 25  # Acoustic Guitar (steel)
    track.isPercussionTrack = False

    # Standard tuning (Guitar Pro uses high-to-low order, already default)
    track.strings = [
        guitarpro.GuitarString(1, 64),   # E4
        guitarpro.GuitarString(2, 59),   # B3
        guitarpro.GuitarString(3, 55),   # G3
        guitarpro.GuitarString(4, 50),   # D3
        guitarpro.GuitarString(5, 45),   # A2
        guitarpro.GuitarString(6, 40),   # E2
    ]

    beats_per_sec = bpm / 60.0
    beats_per_measure = 4
    sec_per_measure = beats_per_measure / beats_per_sec

    if not fretted_notes:
        guitarpro.write(song, output_path)
        return output_path

    max_time = max(n[0] + n[4] for n in fretted_notes)
    num_measures = max(1, math.ceil(max_time / sec_per_measure) + 1)

    # Ensure enough measures exist
    while len(song.measureHeaders) < num_measures:
        header = guitarpro.MeasureHeader()
        song.measureHeaders.append(header)
        for t in song.tracks:
            measure = guitarpro.Measure(t, header)
            t.measures.append(measure)

    # Group notes by onset time
    note_groups = {}
    for i, n in enumerate(fretted_notes):
        t = round(n[0], 3)
        if t not in note_groups:
            note_groups[t] = []
        art = articulations[i] if i < len(articulations) else ArticulationType.PICKED
        note_groups[t].append((n, art))

    # Place notes into measures
    for onset_time in sorted(note_groups.keys()):
        group = note_groups[onset_time]

        measure_idx = int(onset_time / sec_per_measure)
        if measure_idx >= len(track.measures):
            continue

        measure = track.measures[measure_idx]

        # Create a beat
        beat = guitarpro.Beat(measure.voices[0])
        dur_beats = group[0][0][4] * beats_per_sec
        beat.duration = guitarpro.Duration(value=_duration_to_gp(dur_beats))

        for (note_data, art) in group:
            t, note_name, freq, vel, dur, string_idx, fret = note_data

            # Our string_idx: 0=low E(6th), 5=high E(1st)
            # GP string number: 1=high E, 6=low E
            gp_string = 6 - string_idx

            note = guitarpro.Note(beat)
            note.value = fret
            note.string = gp_string
            note.velocity = min(max(vel, 1), 127)

            # Apply articulation
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

    guitarpro.write(song, output_path)
    return output_path
