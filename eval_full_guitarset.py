"""Score Basic Pitch on all 360 GuitarSet tracks and analyze failures."""
import sys; sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')
import mirdata
import mir_eval
import numpy as np
import json
import os
from datetime import datetime
from collections import defaultdict
from basic_pitch.inference import predict
import librosa


def score_track(ref_intervals, ref_pitches_midi, pred_intervals, pred_pitches_midi):
    """Score predictions vs ground truth. Both pitches in MIDI."""
    if len(pred_intervals) == 0 or len(ref_intervals) == 0:
        return {'onset_f1': 0, 'onset_p': 0, 'onset_r': 0,
                'pitch_f1': 0, 'pitch_p': 0, 'pitch_r': 0,
                'full_f1': 0}

    ref_midi = np.round(np.array(ref_pitches_midi, dtype=float))
    pred_midi = np.round(np.array(pred_pitches_midi, dtype=float))

    on_p, on_r, on_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_midi, pred_intervals, pred_midi,
        onset_tolerance=0.05, pitch_tolerance=50, offset_ratio=None)

    op_p, op_r, op_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_midi, pred_intervals, pred_midi,
        onset_tolerance=0.05, pitch_tolerance=0.5, offset_ratio=None)

    _, _, full_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_midi, pred_intervals, pred_midi,
        onset_tolerance=0.05, pitch_tolerance=0.5, offset_ratio=0.2)

    return {
        'onset_f1': round(on_f1, 3), 'onset_p': round(on_p, 3), 'onset_r': round(on_r, 3),
        'pitch_f1': round(op_f1, 3), 'pitch_p': round(op_p, 3), 'pitch_r': round(op_r, 3),
        'full_f1': round(full_f1, 3),
    }


def analyze_pitch_errors(ref_intervals, ref_midi, pred_intervals, pred_midi,
                         onset_tolerance=0.05):
    """Analyze specific pitch error patterns."""
    errors = defaultdict(int)
    matched_ref = set()

    for pi, (ps, pe) in enumerate(pred_intervals):
        pm = pred_midi[pi]
        best_dist = float('inf')
        best_ri = None

        for ri, (rs, re) in enumerate(ref_intervals):
            if ri in matched_ref:
                continue
            if abs(ps - rs) <= onset_tolerance:
                dist = abs(pm - ref_midi[ri])
                if dist < best_dist:
                    best_dist = dist
                    best_ri = ri

        if best_ri is not None and best_dist <= 0.5:
            matched_ref.add(best_ri)
        elif best_ri is not None:
            # Wrong pitch — categorize the error
            diff = int(round(pm - ref_midi[best_ri]))
            if diff == 12 or diff == -12:
                errors['octave_error'] += 1
            elif diff == 7 or diff == -7:
                errors['fifth_error'] += 1
            elif abs(diff) <= 2:
                errors['semitone_error'] += 1
            else:
                errors['large_pitch_error'] += 1
            matched_ref.add(best_ri)
        else:
            errors['extra_note'] += 1

    # Missed notes
    errors['missed_note'] = len(ref_intervals) - len(matched_ref)

    return dict(errors)


def main():
    gs = mirdata.initialize('guitarset', data_home='./data/guitarset')
    all_ids = gs.track_ids
    print(f"Scoring Basic Pitch on {len(all_ids)} GuitarSet tracks...")

    results = []
    by_style = defaultdict(list)
    by_mode = defaultdict(list)
    all_errors = defaultdict(int)
    failed = 0

    for i, tid in enumerate(all_ids):
        track = gs.track(tid)

        if not track.audio_mic_path or not os.path.exists(track.audio_mic_path):
            failed += 1
            continue

        try:
            _, _, events = predict(track.audio_mic_path)
        except Exception as e:
            print(f"  FAIL {tid}: {e}")
            failed += 1
            continue

        pred_intervals = np.array([[e[0], e[1]] for e in events])
        pred_midi = np.array([float(e[2]) for e in events])

        gt = track.notes_all
        if gt is None or len(gt.intervals) == 0:
            failed += 1
            continue

        scores = score_track(gt.intervals, gt.pitches, pred_intervals, pred_midi)
        scores['track_id'] = tid
        scores['style'] = track.style
        scores['mode'] = track.mode
        scores['tempo'] = track.tempo
        scores['gt_notes'] = len(gt.intervals)
        scores['pred_notes'] = len(pred_intervals)
        results.append(scores)

        by_style[track.style].append(scores)
        by_mode[track.mode].append(scores)

        # Analyze pitch errors
        ref_midi_r = np.round(np.array(gt.pitches, dtype=float))
        pred_midi_r = np.round(pred_midi)
        errs = analyze_pitch_errors(gt.intervals, ref_midi_r,
                                     pred_intervals, pred_midi_r)
        for k, v in errs.items():
            all_errors[k] += v

        if (i + 1) % 60 == 0:
            avg_f1 = np.mean([r['pitch_f1'] for r in results])
            print(f"  [{i+1}/{len(all_ids)}] avg pitch F1 so far: {avg_f1:.3f}")

    # ==========================================
    # RESULTS
    # ==========================================
    print(f"\n{'='*70}")
    print(f"BASIC PITCH ON GUITARSET — {len(results)} tracks scored, {failed} failed")
    print(f"{'='*70}")

    avg = lambda key: np.mean([r[key] for r in results])
    print(f"\nOverall Averages:")
    print(f"  Onset F1:       {avg('onset_f1'):.3f} (P={avg('onset_p'):.3f} R={avg('onset_r'):.3f})")
    print(f"  Onset+Pitch F1: {avg('pitch_f1'):.3f} (P={avg('pitch_p'):.3f} R={avg('pitch_r'):.3f})")
    print(f"  Full F1:        {avg('full_f1'):.3f}")

    # By style
    print(f"\nBy Style:")
    print(f"  {'Style':<22s} {'Onset F1':>10s} {'Pitch F1':>10s} {'Full F1':>10s} {'Tracks':>8s}")
    print(f"  {'-'*62}")
    for style in sorted(by_style.keys()):
        tracks = by_style[style]
        print(f"  {style:<22s} {np.mean([r['onset_f1'] for r in tracks]):>10.3f} "
              f"{np.mean([r['pitch_f1'] for r in tracks]):>10.3f} "
              f"{np.mean([r['full_f1'] for r in tracks]):>10.3f} "
              f"{len(tracks):>8d}")

    # By mode
    print(f"\nBy Mode:")
    for mode in sorted(by_mode.keys()):
        tracks = by_mode[mode]
        print(f"  {mode:<10s} Onset={np.mean([r['onset_f1'] for r in tracks]):.3f} "
              f"Pitch={np.mean([r['pitch_f1'] for r in tracks]):.3f} "
              f"Full={np.mean([r['full_f1'] for r in tracks]):.3f} "
              f"({len(tracks)} tracks)")

    # Error analysis
    total_gt = sum(r['gt_notes'] for r in results)
    total_pred = sum(r['pred_notes'] for r in results)
    print(f"\nError Analysis ({total_gt} GT notes, {total_pred} predicted):")
    print(f"  Missed notes:      {all_errors['missed_note']:>6d} ({100*all_errors['missed_note']/total_gt:.1f}% of GT)")
    print(f"  Extra notes:       {all_errors['extra_note']:>6d} ({100*all_errors['extra_note']/total_pred:.1f}% of pred)")
    print(f"  Octave errors:     {all_errors['octave_error']:>6d}")
    print(f"  Fifth errors:      {all_errors['fifth_error']:>6d}")
    print(f"  Semitone errors:   {all_errors['semitone_error']:>6d}")
    print(f"  Large pitch errors:{all_errors['large_pitch_error']:>6d}")

    # Best/worst tracks
    sorted_by_pitch = sorted(results, key=lambda r: r['pitch_f1'])
    print(f"\nTop 5 tracks:")
    for r in sorted_by_pitch[-5:]:
        print(f"  {r['track_id']}: pitch_f1={r['pitch_f1']:.3f} "
              f"({r['style']}, {r['mode']}, {r['tempo']}bpm)")
    print(f"\nBottom 5 tracks:")
    for r in sorted_by_pitch[:5]:
        print(f"  {r['track_id']}: pitch_f1={r['pitch_f1']:.3f} "
              f"({r['style']}, {r['mode']}, {r['tempo']}bpm)")

    # Save full results
    log = {
        'timestamp': datetime.now().isoformat(),
        'engine': 'basic-pitch 0.4.0 (ONNX, default params)',
        'tracks_scored': len(results),
        'tracks_failed': failed,
        'overall': {
            'onset_f1': round(avg('onset_f1'), 3),
            'pitch_f1': round(avg('pitch_f1'), 3),
            'full_f1': round(avg('full_f1'), 3),
        },
        'by_style': {style: {
            'onset_f1': round(np.mean([r['onset_f1'] for r in tracks]), 3),
            'pitch_f1': round(np.mean([r['pitch_f1'] for r in tracks]), 3),
            'full_f1': round(np.mean([r['full_f1'] for r in tracks]), 3),
        } for style, tracks in by_style.items()},
        'by_mode': {mode: {
            'onset_f1': round(np.mean([r['onset_f1'] for r in tracks]), 3),
            'pitch_f1': round(np.mean([r['pitch_f1'] for r in tracks]), 3),
            'full_f1': round(np.mean([r['full_f1'] for r in tracks]), 3),
        } for mode, tracks in by_mode.items()},
        'errors': dict(all_errors),
        'per_track': results,
    }
    os.makedirs('output', exist_ok=True)
    with open('output/full_guitarset_eval.json', 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\nFull results saved to output/full_guitarset_eval.json")


if __name__ == '__main__':
    main()
