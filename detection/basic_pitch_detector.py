"""Basic Pitch (Spotify) note detection for guitar transcription.

Uses the ONNX backend for fast inference. Returns notes in the same
format as librosa_detector for drop-in compatibility with the pipeline.
"""
import numpy as np
import librosa
from basic_pitch.inference import predict


def detect_notes_bp(filepath, min_note_length_ms=50, min_frequency=None,
                    max_frequency=None, onset_threshold=0.5,
                    frame_threshold=0.3):
    """Detect notes using Basic Pitch.

    Args:
        filepath: path to audio file (WAV/MP3)
        min_note_length_ms: minimum note duration in ms
        onset_threshold: sensitivity for note onsets (0-1, lower=more notes)
        frame_threshold: sensitivity for note frames (0-1, lower=more notes)

    Returns:
        list of (time, note_name, freq_hz, velocity, duration)
        Same format as librosa_detector.detect_notes()
    """
    model_output, midi_data, note_events = predict(
        filepath,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_length_ms / 1000.0,
        minimum_frequency=min_frequency,
        maximum_frequency=max_frequency,
    )

    notes = []
    for event in note_events:
        start_time = float(event[0])
        end_time = float(event[1])
        midi_note = int(event[2])
        amplitude = float(event[3])

        duration = end_time - start_time
        note_name = librosa.midi_to_note(midi_note)
        freq_hz = float(librosa.midi_to_hz(midi_note))
        velocity = int(min(127, max(30, amplitude * 127)))

        notes.append((start_time, note_name, freq_hz, velocity, duration))

    # Sort by time
    notes.sort(key=lambda x: x[0])
    return notes


def detect_tempo_bp(filepath):
    """Detect tempo using librosa (Basic Pitch doesn't output tempo)."""
    y, sr = librosa.load(filepath, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(np.round(tempo[0] if hasattr(tempo, '__len__') else tempo))
