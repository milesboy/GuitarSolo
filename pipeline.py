"""GuitarSolo Pipeline — top-level orchestrator.

WAV → Basic Pitch detection → refine → articulations
    → map to frets → export Guitar Pro + MIDI

Usage:
    python pipeline.py                    # interactive song picker
    python pipeline.py song.wav          # direct file
    python pipeline.py --no-optimize     # skip optimization loop
    python pipeline.py --legacy          # use old librosa CQT detector
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import librosa
import numpy as np

from detection.librosa_detector import detect_tempo, detect_key, detect_chords
from export.midi_writer import write_midi
from export.fret_mapper import map_notes_sequence
from export.guitarpro_writer import write_guitarpro
from articulation.detector import detect_articulations


def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def run_pipeline(filepath, optimize=True, verbose=True, use_basic_pitch=True):
    """Run the full analysis pipeline.

    Args:
        filepath: path to WAV/MP3 audio file
        optimize: if True, run parameter grid search (librosa only)
        verbose: print progress
        use_basic_pitch: if True, use Basic Pitch (default); if False, use librosa CQT

    Returns:
        dict with all analysis results
    """
    t_start = time.time()

    if verbose:
        engine = "Basic Pitch" if use_basic_pitch else "Librosa CQT"
        print(f"\n{'='*60}")
        print(f"  GuitarSolo Pipeline ({engine})")
        print(f"  File: {os.path.basename(filepath)}")
        print(f"{'='*60}")

    # --- Load audio for analysis ---
    if verbose:
        print("\n[1/8] Loading audio...", end=" ", flush=True)
    y, sr = librosa.load(filepath, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    if verbose:
        print(f"({format_time(duration)})")

    # --- Basic analysis (tempo, key, chords — always librosa) ---
    if verbose:
        print("[2/8] Analyzing tempo, key, chords...", end=" ", flush=True)
    bpm = detect_tempo(y, sr)
    key = detect_key(y, sr)
    chords = detect_chords(y, sr)
    if verbose:
        print(f"{bpm} BPM, {key}, {len(chords)} chord changes")

    # --- Note detection ---
    if use_basic_pitch:
        use_ensemble = "--ensemble" in sys.argv
        if use_ensemble:
            if verbose:
                print("[3/8] Detecting notes (BP Ensemble)...", flush=True)
            from detection.ensemble_detector import detect_notes_ensemble
            notes = detect_notes_ensemble(filepath, verbose=verbose)
            best_params = {"engine": "basic-pitch-ensemble"}
        else:
            if verbose:
                print("[3/8] Detecting notes (Basic Pitch)...", end=" ", flush=True)
            from detection.basic_pitch_detector import detect_notes_bp
            notes = detect_notes_bp(filepath)
            best_params = {"engine": "basic-pitch"}
            if verbose:
                print(f"{len(notes)} notes")
        final_score = None
    elif optimize:
        if verbose:
            print("[3/8] Running parameter optimization (librosa)...")
        from detection.librosa_detector import detect_notes
        from optimization.grid_search import grid_search, optimize_with_postprocess

        best_params, notes, best_score, _ = grid_search(
            y, sr, bpm, key, filepath, verbose=verbose
        )
        notes, final_score = optimize_with_postprocess(
            y, sr, bpm, key, filepath, notes, verbose=verbose
        )
        if verbose:
            print(f"\n  Optimized: {len(notes)} notes, score={final_score:.3f}")
    else:
        if verbose:
            print("[3/8] Detecting notes (librosa CQT)...", end=" ", flush=True)
        from detection.librosa_detector import detect_notes
        notes = detect_notes(y, sr, bpm)
        best_params = {"engine": "librosa-cqt"}
        final_score = None
        if verbose:
            print(f"{len(notes)} notes")

    # --- Note refinement ---
    if use_basic_pitch:
        if verbose:
            print("[4/8] Post-processing BP output...", flush=True)
        from optimization.bp_postprocess import postprocess_bp
        notes = postprocess_bp(notes, verbose=verbose)
    else:
        if verbose:
            print("[4/8] Refining notes (CQT)...", flush=True)
        from optimization.note_refiner import refine_notes
        notes = refine_notes(y, sr, notes, bpm, verbose=verbose)

    # --- Filter notes above guitar range (E6 = MIDI 88) ---
    max_midi = 88
    before_count = len(notes)
    filtered = []
    for n in notes:
        note_clean = n[1].replace("\u266f", "#").replace("\u266d", "b")
        try:
            midi = librosa.note_to_midi(note_clean)
            if midi <= max_midi:
                filtered.append(n)
        except Exception:
            filtered.append(n)
    notes = filtered
    killed = before_count - len(notes)
    if verbose and killed > 0:
        print(f"  Removed {killed} notes above E6")

    # --- Articulation detection ---
    if verbose:
        print("[5/8] Detecting articulations...", end=" ", flush=True)
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
        print("[6/8] Mapping to fretboard...", end=" ", flush=True)
    fretted = map_notes_sequence(notes)
    if verbose:
        frets_used = set(n[-1] for n in fretted if n[-1] > 0)
        max_fret = max(frets_used) if frets_used else 0
        print(f"max fret: {max_fret}")

    # --- Export Guitar Pro ---
    if verbose:
        print("[7/8] Exporting Guitar Pro...", end=" ", flush=True)
    gp_path = os.path.splitext(filepath)[0] + ".gp5"
    title = os.path.splitext(os.path.basename(filepath))[0]
    title = title.encode("ascii", errors="ignore").decode("ascii")
    title = title[:50]
    write_guitarpro(
        fretted, articulations, bpm, key, chords,
        title=title, output_path=gp_path,
    )
    if verbose:
        print(f"{os.path.basename(gp_path)}")

    # --- Export MIDI ---
    if verbose:
        print("[8/8] Exporting MIDI...", end=" ", flush=True)
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
    use_bp = "--legacy" not in sys.argv
    optimize = "--no-optimize" not in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if args:
        filepath = args[0]
    else:
        from extract import pick_song
        filepath = pick_song()

    run_pipeline(filepath, optimize=optimize, use_basic_pitch=use_bp)
