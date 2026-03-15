"""Aubio-based note detection via CLI subprocess.

Falls back to subprocess calls to aubionotes since the aubio Python
package requires C++ build tools on Windows.
"""
import subprocess
import os
import tempfile

from config import AUBIO_PATH


def _find_aubio():
    """Find aubionotes executable."""
    if os.path.isfile(AUBIO_PATH):
        return AUBIO_PATH
    # Check PATH
    for p in os.environ.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(p, "aubionotes.exe")
        if os.path.isfile(candidate):
            return candidate
        candidate = os.path.join(p, "aubionotes")
        if os.path.isfile(candidate):
            return candidate
    return None


def is_available():
    """Check if aubio CLI tools are available."""
    return _find_aubio() is not None


def detect_notes_aubio(wav_path, buf_size=1024, hop_size=512,
                       onset_threshold=0.3, pitch_method="yinfft",
                       silence_threshold=-40, minioi_ms=30):
    """Detect notes using aubionotes CLI.

    Returns list of (time, midi_pitch, velocity, duration).
    """
    exe = _find_aubio()
    if exe is None:
        raise RuntimeError(
            "aubionotes not found. Install aubio CLI tools or place "
            "aubionotes.exe in .aubio/ directory."
        )

    cmd = [
        exe,
        "-i", wav_path,
        "-B", str(buf_size),
        "-H", str(hop_size),
        "-s", str(silence_threshold),
        "-j", str(minioi_ms),
        "-p", pitch_method,
        "-T", str(onset_threshold),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"aubionotes failed: {result.stderr}")

    notes = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) >= 3:
            midi_pitch = float(parts[0])
            onset_time = float(parts[1])
            duration = float(parts[2])
            if midi_pitch > 0:
                velocity = 80  # aubio CLI doesn't output velocity
                notes.append((onset_time, int(midi_pitch), velocity, duration))

    return notes


def detect_onsets_aubio(wav_path, buf_size=1024, hop_size=512,
                        onset_threshold=0.3, silence_threshold=-40):
    """Detect onset times using aubioonset CLI.

    Returns list of onset times in seconds.
    """
    exe = _find_aubio()
    if exe is None:
        raise RuntimeError("aubio CLI tools not found.")

    # Try aubioonset
    onset_exe = exe.replace("aubionotes", "aubioonset")
    if not os.path.isfile(onset_exe):
        onset_exe = exe  # fall back to aubionotes

    cmd = [
        onset_exe,
        "-i", wav_path,
        "-B", str(buf_size),
        "-H", str(hop_size),
        "-s", str(silence_threshold),
        "-T", str(onset_threshold),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    onsets = []
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            try:
                onsets.append(float(line.strip()))
            except ValueError:
                pass
    return onsets
