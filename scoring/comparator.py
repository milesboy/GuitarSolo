"""Render MIDI to WAV via FluidSynth and compare against original audio."""
import os
import subprocess
import tempfile
import hashlib

from config import FLUIDSYNTH_PATH, SOUNDFONT_PATH
from scoring.metrics import combined_score


def _find_fluidsynth():
    """Find fluidsynth executable."""
    if os.path.isfile(FLUIDSYNTH_PATH):
        return FLUIDSYNTH_PATH
    for p in os.environ.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(p, "fluidsynth.exe")
        if os.path.isfile(candidate):
            return candidate
        candidate = os.path.join(p, "fluidsynth")
        if os.path.isfile(candidate):
            return candidate
    return None


def render_midi_to_wav(midi_path, wav_path=None, soundfont=None):
    """Render a MIDI file to WAV using FluidSynth.

    Returns path to the rendered WAV file.
    """
    exe = _find_fluidsynth()
    if exe is None:
        raise RuntimeError(
            "FluidSynth not found. Place fluidsynth.exe in .fluidsynth/ "
            "or install it to PATH."
        )

    sf = soundfont or SOUNDFONT_PATH
    if not os.path.isfile(sf):
        raise RuntimeError(
            f"Soundfont not found at {sf}. Download FluidR3_GM.sf2 "
            "and place it in soundfonts/ directory."
        )

    if wav_path is None:
        wav_path = os.path.splitext(midi_path)[0] + "_rendered.wav"

    cmd = [
        exe,
        "-ni",           # no interactive mode
        sf,              # soundfont
        midi_path,       # input MIDI
        "-F", wav_path,  # output WAV
        "-r", "44100",   # sample rate
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"FluidSynth render failed: {result.stderr}")

    return wav_path


# Cache rendered WAVs to avoid redundant re-rendering
_render_cache = {}


def compare_midi_to_original(midi_path, original_wav_path, weights=None):
    """Render MIDI to WAV, compare against original, return scores.

    Caches rendered WAVs based on MIDI file content hash.

    Returns:
        dict with individual metric scores and weighted total.
    """
    # Hash MIDI content for caching
    with open(midi_path, "rb") as f:
        midi_hash = hashlib.md5(f.read()).hexdigest()

    if midi_hash in _render_cache and os.path.isfile(_render_cache[midi_hash]):
        rendered_wav = _render_cache[midi_hash]
    else:
        rendered_wav = render_midi_to_wav(midi_path)
        _render_cache[midi_hash] = rendered_wav

    return combined_score(original_wav_path, rendered_wav, weights=weights)


def quick_chroma_score(notes, y_original, sr):
    """Fast chroma-only comparison without rendering.

    Builds a synthetic chroma histogram from detected notes and compares
    against the original audio's chroma. Used as a cheap pre-filter
    in the grid search.

    Args:
        notes: list of (time, note_name, freq, velocity, duration)
        y_original: original audio signal
        sr: sample rate

    Returns:
        float similarity score in [0, 1]
    """
    import librosa
    import numpy as np

    # Build chroma from detected notes
    note_chroma = np.zeros(12)
    for _, note_name, _, velocity, dur in notes:
        # Parse note name to pitch class
        note_clean = note_name.replace("♯", "#").replace("♭", "b")
        try:
            midi = librosa.note_to_midi(note_clean)
            pitch_class = midi % 12
            note_chroma[pitch_class] += velocity * dur
        except Exception:
            continue

    # Get original chroma
    orig_chroma = np.sum(librosa.feature.chroma_cqt(y=y_original, sr=sr), axis=1)

    # Cosine similarity
    norm_n = np.linalg.norm(note_chroma)
    norm_o = np.linalg.norm(orig_chroma)
    if norm_n < 1e-10 or norm_o < 1e-10:
        return 0.0

    return float(np.dot(note_chroma, orig_chroma) / (norm_n * norm_o))
