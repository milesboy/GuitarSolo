"""MIDI export from detected notes and chords."""
import os
import librosa
from midiutil import MIDIFile

from config import NYLON_MIDI_PROGRAM

CHROMA_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def write_midi(filepath, bpm, notes, chords=None):
    """Export detected notes (and optional chords) to a MIDI file.

    Args:
        filepath: Base path for output (extension replaced with .mid)
        bpm: Tempo in BPM
        notes: List of (time, note_name, freq, velocity, duration)
        chords: Optional list of (time, chord_name)

    Returns:
        Path to the written MIDI file.
    """
    num_tracks = 2 if chords else 1
    midi = MIDIFile(num_tracks, deinterleave=False)
    beats_per_sec = bpm / 60.0

    # Track 0: Melody
    midi.addTrackName(0, 0, "Melody")
    midi.addTempo(0, 0, bpm)
    midi.addProgramChange(0, 0, 0, NYLON_MIDI_PROGRAM)

    for time_stamp, note, freq, velocity, dur_sec in notes:
        midi_note = librosa.note_to_midi(note)
        start_beat = time_stamp * beats_per_sec
        dur_beats = max(dur_sec * beats_per_sec, 0.25)
        midi.addNote(0, 0, midi_note, start_beat, dur_beats, velocity)

    # Track 1: Chords (optional)
    if chords:
        midi.addTrackName(1, 0, "Chords")
        midi.addTempo(1, 0, bpm)
        midi.addProgramChange(1, 1, 0, NYLON_MIDI_PROGRAM)

        CHORD_MIDI = {}
        for i, note_name in enumerate(CHROMA_NOTES):
            base = 48 + i
            CHORD_MIDI[note_name] = [base, base + 4, base + 7]
            CHORD_MIDI[f"{note_name}m"] = [base, base + 3, base + 7]

        for i, (time_stamp, chord) in enumerate(chords):
            if chord not in CHORD_MIDI:
                continue
            start_beat = time_stamp * beats_per_sec
            if i + 1 < len(chords):
                dur_sec = chords[i + 1][0] - time_stamp
            else:
                dur_sec = 2.0
            dur_beats = max(dur_sec * beats_per_sec, 0.5)
            for midi_note in CHORD_MIDI[chord]:
                midi.addNote(1, 1, midi_note, start_beat, dur_beats, 80)

    midi_path = os.path.splitext(filepath)[0] + ".mid"
    with open(midi_path, "wb") as f:
        midi.writeFile(f)
    return midi_path
