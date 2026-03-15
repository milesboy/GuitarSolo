"""Parameter optimization via grid search with chroma pre-filtering.

Two-tier approach:
  1. Cheap chroma-only pre-filter over all parameter combos
  2. Full FluidSynth render + multi-metric scoring for top candidates
"""
import itertools
import time

from config import LIBROSA_PARAM_GRID, PREFILTER_TOP_N
from detection.librosa_detector import detect_notes
from scoring.comparator import quick_chroma_score, compare_midi_to_original
from export.midi_writer import write_midi
from optimization.postprocess import apply_all


def _param_combos(grid):
    """Generate all parameter combinations from a grid dict."""
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def grid_search(y, sr, bpm, key, wav_path,
                param_grid=None, top_n=None, verbose=True):
    """Run parameter grid search to find best detection settings.

    Pass 1: Quick chroma scoring (no rendering) over all combos.
    Pass 2: Full render + multi-metric scoring for top N candidates.

    Args:
        y: audio signal
        sr: sample rate
        bpm: detected tempo
        key: detected key (e.g. "A major")
        wav_path: path to original WAV (for full scoring)
        param_grid: dict of parameter ranges (defaults to LIBROSA_PARAM_GRID)
        top_n: number of candidates to fully score (defaults to PREFILTER_TOP_N)

    Returns:
        (best_params, best_notes, best_score, all_results)
    """
    if param_grid is None:
        param_grid = LIBROSA_PARAM_GRID
    if top_n is None:
        top_n = PREFILTER_TOP_N

    combos = list(_param_combos(param_grid))
    total = len(combos)

    if verbose:
        print(f"\n=== PASS 1: Chroma pre-filter ({total} combinations) ===")

    # Pass 1: quick chroma scoring
    chroma_results = []
    t0 = time.time()

    for i, params in enumerate(combos):
        try:
            notes = detect_notes(y, sr, bpm, **params)
            score = quick_chroma_score(notes, y, sr)
            chroma_results.append((score, params, notes))
        except Exception as e:
            if verbose:
                print(f"  [{i+1}/{total}] params={params} -> ERROR: {e}")
            continue

        if verbose and (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate
            print(f"  [{i+1}/{total}] {elapsed:.1f}s elapsed, "
                  f"~{remaining:.0f}s remaining, best={chroma_results[-1][0]:.3f}")

    chroma_results.sort(key=lambda x: x[0], reverse=True)
    candidates = chroma_results[:top_n]

    if verbose:
        elapsed = time.time() - t0
        print(f"  Pass 1 complete: {elapsed:.1f}s, "
              f"top score={candidates[0][0]:.3f}")

    # Pass 2: full render + multi-metric scoring
    # Only if FluidSynth and soundfont are available
    try:
        from scoring.comparator import render_midi_to_wav
        import os
        from config import SOUNDFONT_PATH, FLUIDSYNTH_PATH
        can_render = (os.path.isfile(FLUIDSYNTH_PATH) and
                      os.path.isfile(SOUNDFONT_PATH))
    except Exception:
        can_render = False

    if can_render and verbose:
        print(f"\n=== PASS 2: Full scoring (top {len(candidates)}) ===")

    best_params = candidates[0][1]
    best_notes = candidates[0][2]
    best_total = candidates[0][0]
    all_results = []

    if can_render:
        import tempfile
        import os

        for i, (chroma_score, params, notes) in enumerate(candidates):
            try:
                # Write temp MIDI
                with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
                    tmp_midi = f.name

                write_midi(tmp_midi, bpm, notes)
                scores = compare_midi_to_original(tmp_midi, wav_path)
                total_score = scores["total"]

                all_results.append((total_score, params, notes, scores))

                if total_score > best_total:
                    best_total = total_score
                    best_params = params
                    best_notes = notes

                if verbose:
                    print(f"  [{i+1}/{len(candidates)}] "
                          f"total={total_score:.3f} "
                          f"(chroma={scores['chroma']:.3f} "
                          f"onset={scores['onset']:.3f} "
                          f"spectral={scores['spectral']:.3f})")

                os.unlink(tmp_midi)
            except Exception as e:
                if verbose:
                    print(f"  [{i+1}/{len(candidates)}] render error: {e}")
    else:
        if verbose:
            print("\n  (FluidSynth/soundfont not available — "
                  "using chroma scores only)")
        all_results = [(s, p, n, {"chroma": s, "total": s})
                       for s, p, n in candidates]

    if verbose:
        print(f"\n=== Best params: {best_params} ===")
        print(f"=== Best score: {best_total:.3f} ===")
        print(f"=== Notes detected: {len(best_notes)} ===")

    return best_params, best_notes, best_total, all_results


def optimize_with_postprocess(y, sr, bpm, key, wav_path, notes,
                              verbose=True):
    """Pass 3: Try post-processing steps, keep improvements.

    Applies each post-processing step independently, scores the result,
    and keeps only steps that improve the score.
    """
    from scoring.comparator import quick_chroma_score

    base_score = quick_chroma_score(notes, y, sr)
    best_notes = notes
    best_score = base_score

    if verbose:
        print(f"\n=== PASS 3: Post-processing optimization ===")
        print(f"  Baseline score: {base_score:.3f} ({len(notes)} notes)")

    steps = [
        ("filter_outliers", lambda n: apply_all(n, bpm, key,
            quantize=False, filter_short=True, snap_key=False, merge=False)),
        ("merge_repeated", lambda n: apply_all(n, bpm, key,
            quantize=False, filter_short=False, snap_key=False, merge=True)),
        ("snap_to_key", lambda n: apply_all(n, bpm, key,
            quantize=False, filter_short=False, snap_key=True, merge=False)),
        ("quantize", lambda n: apply_all(n, bpm, key,
            quantize=True, filter_short=False, snap_key=False, merge=False)),
    ]

    for step_name, step_fn in steps:
        try:
            candidate = step_fn(best_notes)
            score = quick_chroma_score(candidate, y, sr)
            if score >= best_score:
                if verbose:
                    print(f"  {step_name}: {score:.3f} "
                          f"({len(candidate)} notes) — KEPT")
                best_notes = candidate
                best_score = score
            else:
                if verbose:
                    print(f"  {step_name}: {score:.3f} "
                          f"({len(candidate)} notes) — reverted")
        except Exception as e:
            if verbose:
                print(f"  {step_name}: error — {e}")

    if verbose:
        print(f"  Final: {best_score:.3f} ({len(best_notes)} notes)")

    return best_notes, best_score
