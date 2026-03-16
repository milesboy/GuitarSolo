"""Automated feedback loop: detect → render → compare → fix → repeat.

Iteratively improves transcription by:
1. Rendering current MIDI to WAV
2. Comparing against original audio spectrally
3. Removing extra notes (in MIDI but not in original)
4. Adding missing notes (in original but not in MIDI)
5. Re-rendering and re-comparing until convergence
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import librosa
import json
from datetime import datetime

from scoring.spectral_diff import find_spectral_differences, render_midi_to_wav, compute_cqt_db
from export.midi_writer import write_midi
from config import FLUIDSYNTH_PATH, SOUNDFONT_PATH


def remove_extra_notes(notes, extra_list, tolerance=0.3):
    """Remove notes that the spectral diff identified as extras.

    Matches by time (within tolerance) and pitch.
    """
    removed = 0
    kept = []

    for n in notes:
        t, name, freq, vel, dur = n
        clean = name.replace("\u266f", "#").replace("\u266d", "b")
        try:
            midi = librosa.note_to_midi(clean)
        except Exception:
            kept.append(n)
            continue

        is_extra = False
        for extra in extra_list:
            if abs(t - extra['time']) <= tolerance and midi == extra['midi']:
                is_extra = True
                break

        if is_extra:
            removed += 1
        else:
            kept.append(n)

    return kept, removed


def add_missing_notes(notes, missing_list, y, sr, tolerance=0.3,
                      min_deficit_db=45.0):
    """Add notes that the spectral diff identified as missing.

    Only adds notes with strong deficit (clearly audible in original
    but absent in our MIDI). Uses CQT to verify the note is real
    and estimate duration.
    """
    added = []
    hop = 512

    # Build CQT for duration estimation
    y_harm, _ = librosa.effects.hpss(y)
    tuning = librosa.estimate_tuning(y=y_harm, sr=sr)
    fmin = librosa.note_to_hz("E2") * (2 ** (tuning / 12))
    n_semi = 52
    cqt_raw = np.abs(librosa.cqt(
        y=y_harm, sr=sr, fmin=fmin, hop_length=hop,
        n_bins=n_semi * 3, bins_per_octave=36))
    cqt = np.zeros((n_semi, cqt_raw.shape[1]))
    for i in range(n_semi):
        cqt[i] = np.max(cqt_raw[i*3:(i+1)*3], axis=0)

    # Filter: only add strong missing notes not already in our list
    existing = set()
    for n in notes:
        clean = n[1].replace("\u266f", "#").replace("\u266d", "b")
        try:
            midi = librosa.note_to_midi(clean)
            existing.add((round(n[0] / tolerance), midi))
        except Exception:
            pass

    GUITAR_MIDI_LOW = 40   # E2 — lowest guitar note
    GUITAR_MIDI_HIGH = 88  # E6 — highest harmonic
    BASS_MIDI_MAX = 55     # G3 — bass range ceiling

    for missing in missing_list:
        t = missing['time']
        midi = missing['midi']

        # Hard floor: nothing below guitar range
        if midi < GUITAR_MIDI_LOW or midi > GUITAR_MIDI_HIGH:
            continue

        # Bass notes need a higher deficit threshold — low frequencies
        # have more CQT energy from room tone and string resonance,
        # causing false "missing" detections
        if midi <= BASS_MIDI_MAX:
            required_db = min_deficit_db + 15.0  # 60dB for bass
        else:
            required_db = min_deficit_db

        if missing['deficit_db'] < required_db:
            continue

        key = (round(t / tolerance), midi)
        if key in existing:
            continue

        # Verify with CQT: is there real energy at this pitch/time?
        note_bin = midi - 40
        if note_bin < 0 or note_bin >= n_semi:
            continue

        frame = librosa.time_to_frames(t, sr=sr, hop_length=hop)
        if frame >= cqt.shape[1]:
            continue

        window = max(1, int(0.1 * sr / hop))
        energy = float(np.mean(cqt[note_bin, frame:min(frame+window, cqt.shape[1])]))

        # Must have real CQT energy
        all_mags = cqt[cqt > 0]
        threshold = np.percentile(all_mags, 50) if len(all_mags) > 0 else 0
        if energy < threshold:
            continue

        # Estimate duration from CQT
        dur = 0.1
        check = frame + int(0.1 * sr / hop)
        while check < cqt.shape[1]:
            e = float(np.mean(cqt[note_bin, check:min(check+window, cqt.shape[1])]))
            if e < energy * 0.2:
                break
            dur += 0.1
            check += int(0.1 * sr / hop)
            if dur > 4.0:
                break

        name = librosa.midi_to_note(midi)
        freq = float(librosa.midi_to_hz(midi))
        vel = 70  # moderate default
        added.append((t, name, freq, vel, dur))
        existing.add(key)

    all_notes = sorted(notes + added, key=lambda n: n[0])
    return all_notes, len(added)


def run_feedback_loop(original_wav, notes, bpm, chords=None,
                      max_iterations=5, verbose=True):
    """Run the iterative feedback loop.

    Args:
        original_wav: path to original audio file
        notes: initial note list from BP + postprocessing
        bpm: tempo
        chords: chord list (for MIDI export)
        max_iterations: max improvement cycles
        verbose: print progress

    Returns:
        improved notes, iteration log
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Feedback Loop")
        print(f"  Starting with {len(notes)} notes, max {max_iterations} iterations")
        print(f"{'='*60}")

    y, sr = librosa.load(original_wav, sr=None)
    log = []

    for iteration in range(max_iterations):
        if verbose:
            print(f"\n--- Iteration {iteration + 1} ---")

        # Write current notes to MIDI
        tmp_midi = original_wav.replace('.wav', f'_loop_{iteration}.mid')
        write_midi(tmp_midi, bpm, notes, chords)

        # Compare against original
        if verbose:
            print(f"  Notes: {len(notes)}")
        diff = find_spectral_differences(original_wav, tmp_midi, verbose=verbose)

        entry = {
            'iteration': iteration + 1,
            'notes_before': len(notes),
            'similarity': diff['similarity'],
            'n_extra': diff['n_extra'],
            'n_missing': diff['n_missing'],
        }

        # Remove extras
        notes, n_removed = remove_extra_notes(notes, diff['extra_notes'])
        if verbose:
            print(f"  Removed {n_removed} extra notes")

        # Add missing (CQT-verified)
        notes, n_added = add_missing_notes(
            notes, diff['missing_notes'], y, sr)
        if verbose:
            print(f"  Added {n_added} missing notes")

        entry['removed'] = n_removed
        entry['added'] = n_added
        entry['notes_after'] = len(notes)
        log.append(entry)

        # Clean up temp MIDI
        try:
            os.unlink(tmp_midi)
        except Exception:
            pass

        # Convergence: stop if no changes
        if n_removed == 0 and n_added == 0:
            if verbose:
                print(f"  Converged — no changes to make")
            break

        # Convergence: stop if similarity isn't improving
        if len(log) >= 2:
            if log[-1]['similarity'] <= log[-2]['similarity'] - 0.01:
                if verbose:
                    print(f"  Similarity dropped — stopping")
                break

    # Final score
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Feedback Loop Complete")
        print(f"  Iterations: {len(log)}")
        print(f"  Notes: {log[0]['notes_before']} → {len(notes)}")
        if log:
            print(f"  Similarity: {log[0]['similarity']:.1%} → {log[-1]['similarity']:.1%}")
        print(f"{'='*60}")

    return notes, log


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    import warnings; warnings.filterwarnings('ignore')
    from basic_pitch.inference import predict
    from optimization.bp_postprocess import postprocess_bp
    from detection.librosa_detector import detect_tempo, detect_key, detect_chords

    wav = sys.argv[1] if len(sys.argv) > 1 else 'downloads/Kelly Valleau - In My Life (The Beatles) - Fingerstyle Guitar.wav'

    print("Running Basic Pitch...", end=" ", flush=True)
    _, _, events = predict(wav)
    bp_notes = [(float(e[0]), librosa.midi_to_note(int(e[2])),
                 float(librosa.midi_to_hz(int(e[2]))), int(e[3]*127),
                 float(e[1]-e[0])) for e in events]
    print(f"{len(bp_notes)} notes")

    print("Post-processing...")
    notes = postprocess_bp(bp_notes, verbose=True)

    y, sr = librosa.load(wav, sr=None)
    bpm = detect_tempo(y, sr)
    chords = detect_chords(y, sr)

    improved, log = run_feedback_loop(wav, notes, bpm, chords,
                                       max_iterations=5)

    # Write final MIDI
    midi_path = wav.replace('.wav', '.mid')
    write_midi(midi_path, bpm, improved, chords)
    print(f"\nFinal MIDI: {midi_path} ({len(improved)} notes)")

    # Save log
    os.makedirs('output', exist_ok=True)
    with open('output/feedback_loop_log.json', 'w') as f:
        json.dump(log, f, indent=2)
