"""Detect guitar articulations from audio transitions.

Classifies note transitions as:
  - PICKED: normal plucked note (onset spike present)
  - HAMMER_ON: ascending pitch jump without onset
  - PULL_OFF: descending pitch jump without onset
  - SLIDE_UP: smooth ascending pitch glide between notes
  - SLIDE_DOWN: smooth descending pitch glide between notes
  - BEND: gradual pitch rise within a single note
"""
import librosa
import numpy as np
from enum import Enum


class ArticulationType(Enum):
    PICKED = "picked"
    HAMMER_ON = "hammer_on"
    PULL_OFF = "pull_off"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"
    BEND = "bend"


def _detect_onsets_at_times(y, sr, times, tolerance_ms=20):
    """Check which note start times coincide with detected onsets."""
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    tolerance = tolerance_ms / 1000.0

    has_onset = {}
    for t in times:
        has_onset[t] = any(abs(t - ot) <= tolerance for ot in onset_times)
    return has_onset


def _get_pitch_track(y, sr, t_start, t_end, hop_length=256):
    """Get per-frame pitch track between two times."""
    start_sample = int(t_start * sr)
    end_sample = min(int(t_end * sr), len(y))
    if end_sample <= start_sample:
        return np.array([])

    segment = y[start_sample:end_sample]
    pitches, magnitudes = librosa.piptrack(
        y=segment, sr=sr, hop_length=hop_length,
        fmin=librosa.note_to_hz("E2"),
        fmax=librosa.note_to_hz("E6"),
    )

    # Extract dominant pitch per frame
    pitch_track = []
    for frame in range(pitches.shape[1]):
        idx = magnitudes[:, frame].argmax()
        pitch = pitches[idx, frame]
        if pitch > 0:
            pitch_track.append(pitch)
        else:
            pitch_track.append(0.0)

    return np.array(pitch_track)


def _is_smooth_glide(pitch_track, min_duration_frames=5):
    """Check if pitch track shows a smooth glide (not a jump)."""
    if len(pitch_track) < min_duration_frames:
        return False

    # Filter out zero-pitch frames
    valid = pitch_track[pitch_track > 0]
    if len(valid) < min_duration_frames:
        return False

    # Check if pitch changes gradually (small frame-to-frame deltas)
    deltas = np.abs(np.diff(librosa.hz_to_midi(valid)))
    # A slide has consistent small movements; a jump has one large delta
    # Slide: most deltas < 1 semitone, but total range > 1 semitone
    total_range = abs(librosa.hz_to_midi(valid[-1]) - librosa.hz_to_midi(valid[0]))
    max_delta = np.max(deltas) if len(deltas) > 0 else 0

    return total_range > 1.0 and max_delta < 2.0


def _is_bend(pitch_track, nominal_midi, min_rise=0.5):
    """Check if a note bends up from its nominal pitch."""
    valid = pitch_track[pitch_track > 0]
    if len(valid) < 5:
        return False

    midi_track = librosa.hz_to_midi(valid)
    # Bend: starts near nominal, rises above it
    start_pitch = np.mean(midi_track[:3])
    peak_pitch = np.max(midi_track)

    return (abs(start_pitch - nominal_midi) < 0.5 and
            peak_pitch - nominal_midi > min_rise)


def detect_articulations(y, sr, notes):
    """Classify articulations for each note based on audio analysis.

    Args:
        y: audio signal
        sr: sample rate
        notes: list of (time, note_name, freq, velocity, duration)

    Returns:
        List of ArticulationType, one per note.
    """
    if not notes:
        return []

    # Detect onsets
    note_times = [n[0] for n in notes]
    has_onset = _detect_onsets_at_times(y, sr, note_times)

    articulations = []

    for i, (t, note_name, freq, vel, dur) in enumerate(notes):
        # Default: picked
        art = ArticulationType.PICKED

        # Check for bend within this note
        note_clean = note_name.replace("♯", "#").replace("♭", "b")
        try:
            nominal_midi = librosa.note_to_midi(note_clean)
        except Exception:
            articulations.append(art)
            continue

        pitch_track = _get_pitch_track(y, sr, t, t + dur)
        if len(pitch_track) > 0 and _is_bend(pitch_track, nominal_midi):
            art = ArticulationType.BEND
            articulations.append(art)
            continue

        # Check transition from previous note
        if i > 0 and not has_onset.get(t, True):
            prev_t, prev_note, prev_freq, _, prev_dur = notes[i - 1]
            prev_end = prev_t + prev_dur

            # Check if there's a smooth glide between notes
            if prev_end >= t - 0.05:  # overlapping or nearly adjacent
                transition_track = _get_pitch_track(
                    y, sr,
                    max(prev_end - 0.05, prev_t),
                    min(t + 0.05, t + dur)
                )

                if len(transition_track) > 0 and _is_smooth_glide(transition_track):
                    if freq > prev_freq:
                        art = ArticulationType.SLIDE_UP
                    else:
                        art = ArticulationType.SLIDE_DOWN
                else:
                    # Jump without onset = hammer-on or pull-off
                    if freq > prev_freq:
                        art = ArticulationType.HAMMER_ON
                    else:
                        art = ArticulationType.PULL_OFF

        articulations.append(art)

    return articulations
