"""Central configuration for the GuitarSolo pipeline."""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Paths ---
SOUNDFONT_PATH = os.path.join(PROJECT_ROOT, "soundfonts", "FluidR3_GM.sf2")
FLUIDSYNTH_PATH = os.path.join(PROJECT_ROOT, ".fluidsynth", "fluidsynth.exe")
AUBIO_PATH = os.path.join(PROJECT_ROOT, ".aubio", "aubionotes.exe")

# --- Guitar tuning (standard) ---
# Open string MIDI note numbers: E2, A2, D3, G3, B3, E4
GUITAR_STRINGS = [40, 45, 50, 55, 59, 64]
NUM_FRETS = 24
GUITAR_MIDI_RANGE = (40, 88)  # E2 to E6

# --- Detection parameters (librosa) ---
LIBROSA_PARAM_GRID = {
    "hop_length": [256, 512, 1024],
    "strong_thresh_pct": [0.15, 0.20, 0.25],
    "melody_thresh_pct": [0.08, 0.10, 0.15],
    "sustain_ratio": [0.15, 0.20, 0.30],
    "group_window": [0.06, 0.08, 0.10],
}

# --- Detection parameters (aubio, for when available) ---
AUBIO_PARAM_GRID = {
    "buf_size": [512, 1024, 2048, 4096],
    "hop_size": [128, 256, 512],
    "onset_threshold": [0.1, 0.2, 0.3, 0.5, 0.7],
    "pitch_method": ["yinfft", "yin", "yinfast"],
    "silence_threshold": [-30, -40, -50, -60],
    "minioi_ms": [20, 30, 50, 80],
}

# --- Scoring weights ---
SCORING_WEIGHTS = {
    "chroma": 0.35,
    "onset": 0.25,
    "spectral": 0.20,
    "pitch_histogram": 0.20,
}

# --- Optimization ---
PREFILTER_TOP_N = 50       # chroma-only pre-filter keeps top N combos
MAX_GRID_ITERATIONS = 500  # safety cap on grid search
ONSET_TOLERANCE_MS = 50    # ms tolerance for onset alignment scoring

# --- Post-processing ---
MIN_NOTE_DURATION_MS = 30
MIN_NOTE_VELOCITY = 20
DEFAULT_QUANTIZE_SUBDIVISION = 16  # snap to 16th notes

# --- MIDI ---
GUITAR_MIDI_PROGRAM = 25  # GM: Acoustic Guitar (steel)
NYLON_MIDI_PROGRAM = 24   # GM: Acoustic Guitar (nylon)
