"""GuitarSolo Pipeline — top-level orchestrator.

WAV → detect notes → optimize parameters → detect articulations
    → map to frets → export Guitar Pro + MIDI

Usage:
    python pipeline.py                    # interactive song picker
    python pipeline.py song.wav          # direct file
    python pipeline.py --no-optimize     # skip optimization loop
"""
import sys
import os
import time

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import librosa
import numpy as np

from detection.librosa_detector import detect_tempo, detect_key, detect_chords, detect_notes
from export.midi_writer import write_midi
from export.fret_mapper import map_notes_sequence
from export.guitarpro_writer import write_guitarpro
from articulation.detector import detect_articulations


def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def run_pipeline(filepath, optimize=True, verbose=True):
    """Run the full analysis pipeline.

    Args:
        filepath: path to WAV/MP3 audio file
        optimize: if True, run parameter grid search
        verbose: print progress

    Returns:
        dict with all analysis results
    """
    t_start = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  GuitarSolo Pipeline")
        print(f"  File: {os.path.basename(filepath)}")
        print(f"{'='*60}")

    # --- Load audio ---
    if verbose:
        print("\n[1/7] Loading audio...", end=" ", flush=True)
    y, sr = librosa.load(filepath, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    if verbose:
        print(f"({format_time(duration)})")

    # --- Basic analysis ---
    if verbose:
        print("[2/7] Analyzing tempo, key, chords...", end=" ", flush=True)
    bpm = detect_tempo(y, sr)
    key = detect_key(y, sr)
    chords = detect_chords(y, sr)
    if verbose:
        print(f"{bpm} BPM, {key}, {len(chords)} chord changes")

    # --- Note detection ---
    if optimize:
        if verbose:
            print("[3/7] Running parameter optimization...")

        from optimization.grid_search import grid_search, optimize_with_postprocess

        best_params, notes, best_score, _ = grid_search(
            y, sr, bpm, key, filepath, verbose=verbose
        )

        # Post-processing pass
        notes, final_score = optimize_with_postprocess(
            y, sr, bpm, key, filepath, notes, verbose=verbose
        )

        if verbose:
            print(f"\n  Optimized: {len(notes)} notes, "
                  f"score={final_score:.3f}")
    else:
        if verbose:
            print("[3/7] Detecting notes (single pass)...", end=" ", flush=True)
        notes = detect_notes(y, sr, bpm)
        best_params = {}
        final_score = None
        if verbose:
            print(f"{len(notes)} notes")

    # --- Articulation detection ---
    if verbose:
        print("[4/7] Detecting articulations...", end=" ", flush=True)
    articulations = detect_articulations(y, sr, notes)
    art_counts = {}
    for a in articulations:
        art_counts[a.value] = art_counts.get(a.value, 0) + 1
    if verbose:
        parts = [f"{v} {k}" for k, v in sorted(art_counts.items(),
                                                 key=lambda x: -x[1])]
        print(", ".join(parts))

    # --- Fret mapping ---
    if verbose:
        print("[5/7] Mapping to fretboard...", end=" ", flush=True)
    fretted = map_notes_sequence(notes)
    if verbose:
        frets_used = set(n[-1] for n in fretted if n[-1] > 0)
        max_fret = max(frets_used) if frets_used else 0
        print(f"max fret: {max_fret}")

    # --- Export Guitar Pro ---
    if verbose:
        print("[6/7] Exporting Guitar Pro...", end=" ", flush=True)
    gp_path = os.path.splitext(filepath)[0] + ".gp5"
    title = os.path.splitext(os.path.basename(filepath))[0]
    write_guitarpro(
        fretted, articulations, bpm, key, chords,
        title=title, output_path=gp_path,
    )
    if verbose:
        print(f"{os.path.basename(gp_path)}")

    # --- Export MIDI ---
    if verbose:
        print("[7/7] Exporting MIDI...", end=" ", flush=True)
    midi_path = write_midi(filepath, bpm, notes, chords)
    if verbose:
        print(f"{os.path.basename(midi_path)}")

    elapsed = time.time() - t_start

    # --- Summary ---
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Song:     {title}")
        print(f"  Tempo:    {bpm} BPM")
        print(f"  Key:      {key}")
        print(f"  Length:   {format_time(duration)}")
        print(f"  Notes:    {len(notes)}")
        print(f"  Chords:   {len(chords)} changes")
        if final_score is not None:
            print(f"  Score:    {final_score:.1%}")
        print(f"  Time:     {elapsed:.1f}s")
        print(f"  Output:   {os.path.basename(gp_path)}")
        print(f"            {os.path.basename(midi_path)}")
        print(f"{'='*60}")

        # Chord progression
        print("\n  Chord Progression:")
        for ts, chord in chords:
            print(f"    {format_time(ts):>5s}  {chord}")

        # Notes with articulations
        print("\n  Notes:")
        for i, n in enumerate(fretted):
            t, note, freq, vel, dur = n[:5]
            string, fret = n[5], n[6]
            art = articulations[i] if i < len(articulations) else None
            art_str = f" [{art.value}]" if art and art.value != "picked" else ""
            bar = "#" * (vel // 10)
            print(f"    {format_time(t):>5s}  {note:<5s} "
                  f"s{6-string}f{fret:<2d} vel:{vel:>3d} "
                  f"dur:{dur:.2f}s{art_str}  {bar}")

    return {
        "bpm": bpm,
        "key": key,
        "duration": duration,
        "chords": chords,
        "notes": notes,
        "fretted_notes": fretted,
        "articulations": articulations,
        "midi": midi_path,
        "guitarpro": gp_path,
        "score": final_score,
        "params": best_params,
    }


if __name__ == "__main__":
    optimize = "--no-optimize" not in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if args:
        filepath = args[0]
    else:
        from extract import pick_song
        filepath = pick_song()

    run_pipeline(filepath, optimize=optimize)
