"""Ensemble detector: runs Basic Pitch with multiple parameter configs
and uses voting to determine the most likely correct notes.

Notes that appear across most configs are confident detections.
Notes that only appear with specific parameters are likely artifacts.
Pitch conflicts at the same onset are resolved by vote count + velocity.
"""
import numpy as np
import librosa
from basic_pitch.inference import predict
from collections import defaultdict


# Parameter configs to vote across
BP_CONFIGS = [
    {'onset_threshold': 0.3, 'frame_threshold': 0.3},
    {'onset_threshold': 0.4, 'frame_threshold': 0.3},
    {'onset_threshold': 0.5, 'frame_threshold': 0.3},  # BP default
    {'onset_threshold': 0.5, 'frame_threshold': 0.5},
    {'onset_threshold': 0.6, 'frame_threshold': 0.3},
    {'onset_threshold': 0.7, 'frame_threshold': 0.5},
]

# Minimum vote fraction to keep a note (e.g. 0.5 = must appear in 50%+ of configs)
MIN_VOTE_FRACTION = 0.5

# Onset grouping tolerance
ONSET_TOLERANCE = 0.05  # 50ms


def detect_notes_ensemble(filepath, min_votes=None, configs=None, verbose=True):
    """Detect notes using ensemble of BP configurations.

    Args:
        filepath: path to audio file
        min_votes: minimum number of configs that must detect a note (default: 50%)
        configs: list of BP parameter dicts (default: BP_CONFIGS)
        verbose: print progress

    Returns:
        list of (time, note_name, freq, velocity, duration)
    """
    if configs is None:
        configs = BP_CONFIGS
    n_configs = len(configs)

    if min_votes is None:
        min_votes = max(1, int(n_configs * MIN_VOTE_FRACTION))

    if verbose:
        print(f"  Running {n_configs} BP configs (min votes: {min_votes})...",
              end=" ", flush=True)

    # Collect all detections from all configs
    all_detections = []  # (time, end_time, midi, velocity, config_idx)

    for ci, cfg in enumerate(configs):
        _, _, events = predict(filepath, **cfg)
        for e in events:
            all_detections.append((
                float(e[0]),      # onset
                float(e[1]),      # offset
                int(e[2]),        # midi
                float(e[3]),      # velocity
                ci,               # config index
            ))

    if verbose:
        print(f"{len(all_detections)} total detections")

    # Group detections by onset time (within tolerance)
    all_detections.sort(key=lambda x: x[0])
    onset_groups = []
    if all_detections:
        current = [all_detections[0]]
        for d in all_detections[1:]:
            if d[0] - current[0][0] <= ONSET_TOLERANCE:
                current.append(d)
            else:
                onset_groups.append(current)
                current = [d]
        onset_groups.append(current)

    # Vote on each onset group
    notes = []
    for group in onset_groups:
        # Count votes per MIDI pitch
        pitch_votes = defaultdict(lambda: {
            'configs': set(),
            'total_vel': 0.0,
            'total_dur': 0.0,
            'count': 0,
            'onset_sum': 0.0,
        })

        for onset, offset, midi, vel, ci in group:
            pv = pitch_votes[midi]
            pv['configs'].add(ci)
            pv['total_vel'] += vel
            pv['total_dur'] += (offset - onset)
            pv['count'] += 1
            pv['onset_sum'] += onset

        # Keep pitches that meet the vote threshold
        for midi, pv in pitch_votes.items():
            n_votes = len(pv['configs'])
            if n_votes >= min_votes:
                avg_onset = pv['onset_sum'] / pv['count']
                avg_vel = pv['total_vel'] / pv['count']
                avg_dur = pv['total_dur'] / pv['count']

                name = librosa.midi_to_note(midi)
                freq = float(librosa.midi_to_hz(midi))
                velocity = int(min(127, max(30, avg_vel * 127)))

                notes.append((avg_onset, name, freq, velocity, avg_dur))

    # Sort by time
    notes.sort(key=lambda x: x[0])

    if verbose:
        print(f"  Ensemble result: {len(notes)} notes "
              f"(from {len(onset_groups)} onset groups)")

    return notes


def detect_tempo_ensemble(filepath):
    """Detect tempo using librosa."""
    y, sr = librosa.load(filepath, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(np.round(tempo[0] if hasattr(tempo, '__len__') else tempo))
