"""Compare transcription methods on GuitarSet benchmark."""
import sys; sys.path.insert(0, '.')
import mirdata
import mir_eval
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

import librosa
from detection.librosa_detector import detect_notes, detect_tempo
from basic_pitch.inference import predict

gs = mirdata.initialize('guitarset', data_home='./data/guitarset')

# Same 5 tracks as baseline
styles = ['Bossa Nova', 'Funk', 'Jazz', 'Rock', 'Singer-Songwriter']
selected = []
for style in styles:
    for tid in gs.track_ids:
        t = gs.track(tid)
        if t.mode == 'solo' and t.style == style:
            selected.append(tid)
            break


def score_notes(ref_intervals, ref_pitches_midi, pred_intervals, pred_pitches_midi):
    """Score predictions vs ground truth. Both pitches must be in MIDI."""
    if len(pred_intervals) == 0 or len(ref_intervals) == 0:
        return {'onset_f1': 0, 'onset_pitch_f1': 0, 'full_f1': 0}
    # Round GT pitches to nearest integer MIDI — GuitarSet uses continuous
    # pitch values (e.g. 56.18 instead of 56) which causes false mismatches
    # with integer-MIDI predictions at 0.5 semitone tolerance
    ref_midi = np.round(np.array(ref_pitches_midi, dtype=float))
    pred_midi = np.round(np.array(pred_pitches_midi, dtype=float))

    _, _, on_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_midi, pred_intervals, pred_midi,
        onset_tolerance=0.05, pitch_tolerance=50, offset_ratio=None)
    op_p, op_r, op_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_midi, pred_intervals, pred_midi,
        onset_tolerance=0.05, pitch_tolerance=0.5, offset_ratio=None)
    _, _, full_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_midi, pred_intervals, pred_midi,
        onset_tolerance=0.05, pitch_tolerance=0.5, offset_ratio=0.2)
    return {
        'onset_f1': round(on_f1, 3),
        'onset_pitch_f1': round(op_f1, 3),
        'onset_pitch_p': round(op_p, 3),
        'onset_pitch_r': round(op_r, 3),
        'full_f1': round(full_f1, 3),
    }


# ============================================
# Method 1: Our librosa CQT detector
# ============================================
print("=" * 60)
print("METHOD 1: Our Librosa CQT Detector")
print("=" * 60)

our_results = []
for tid in selected:
    track = gs.track(tid)
    print(f'  {tid} ({track.style})...', end=' ', flush=True)
    y, sr = librosa.load(track.audio_mic_path, sr=None)
    bpm = track.tempo or detect_tempo(y, sr)
    notes = detect_notes(y, sr, bpm)

    pred_intervals = np.array([[n[0], n[0] + n[4]] for n in notes])
    pred_midi = np.array([librosa.hz_to_midi(n[2]) for n in notes])  # convert Hz to MIDI

    gt = track.notes_all  # gt.pitches is already MIDI
    scores = score_notes(gt.intervals, gt.pitches, pred_intervals, pred_midi)
    scores['track_id'] = tid
    scores['style'] = track.style
    scores['pred_notes'] = len(notes)
    our_results.append(scores)
    print(f'{len(notes)} notes  onset={scores["onset_f1"]:.3f}  '
          f'pitch={scores["onset_pitch_f1"]:.3f}  full={scores["full_f1"]:.3f}')


# ============================================
# Method 2: Hybrid — Basic Pitch onsets + CQT pitch
# ============================================
print("\n" + "=" * 60)
print("METHOD 2: Hybrid (Basic Pitch onsets + CQT pitch)")
print("=" * 60)

hybrid_results = []
for tid in selected:
    track = gs.track(tid)
    print(f'  {tid} ({track.style})...', end=' ', flush=True)

    _, _, bp_events = predict(track.audio_mic_path)
    bp_onsets = sorted(set(round(e[0], 3) for e in bp_events))

    y, sr = librosa.load(track.audio_mic_path, sr=None)
    y_harm, _ = librosa.effects.hpss(y)
    tuning = librosa.estimate_tuning(y=y_harm, sr=sr)
    fmin = librosa.note_to_hz("E2") * (2 ** (tuning / 12))
    fmin_std = librosa.note_to_hz("E2")

    hop = 512
    n_semi = 52
    cqt_raw = np.abs(librosa.cqt(y=y_harm, sr=sr, fmin=fmin, hop_length=hop,
        n_bins=n_semi*3, bins_per_octave=36))
    cqt = np.zeros((n_semi, cqt_raw.shape[1]))
    for i in range(n_semi):
        cqt[i] = np.max(cqt_raw[i*3:(i+1)*3], axis=0)
    bf = [float(librosa.midi_to_hz(librosa.hz_to_midi(fmin_std)+i))
          for i in range(n_semi)]

    window = max(1, int(0.15 * sr / hop))
    all_mags = cqt[cqt > 0]
    threshold = np.percentile(all_mags, 50) if len(all_mags) > 0 else 0

    hybrid_notes = []
    for onset_t in bp_onsets:
        frame = librosa.time_to_frames(onset_t, sr=sr, hop_length=hop)
        end = min(frame + window, cqt.shape[1])
        if frame >= cqt.shape[1]:
            continue
        mag = np.mean(cqt[:, frame:end], axis=1)
        max_mag = np.max(mag)
        if max_mag < threshold:
            continue
        strong = max(max_mag * 0.20, threshold)
        for b in range(n_semi):
            if mag[b] < strong:
                continue
            left = mag[b-1] if b > 0 else 0
            right = mag[b+1] if b < n_semi-1 else 0
            if mag[b] > left and mag[b] > right:
                hybrid_notes.append((onset_t, bf[b], 0.2))

    pred_intervals = np.array([[n[0], n[0] + n[2]] for n in hybrid_notes])
    pred_midi = np.array([librosa.hz_to_midi(n[1]) for n in hybrid_notes])  # Hz to MIDI

    gt = track.notes_all  # gt.pitches is MIDI
    scores = score_notes(gt.intervals, gt.pitches, pred_intervals, pred_midi)
    scores['track_id'] = tid
    scores['style'] = track.style
    scores['pred_notes'] = len(hybrid_notes)
    hybrid_results.append(scores)
    print(f'{len(hybrid_notes)} notes  onset={scores["onset_f1"]:.3f}  '
          f'pitch={scores["onset_pitch_f1"]:.3f}  full={scores["full_f1"]:.3f}')


# ============================================
# Method 3: Basic Pitch with octave shift
# ============================================
print("\n" + "=" * 60)
print("METHOD 3: Basic Pitch + Octave Shift (-12)")
print("=" * 60)

bp_shift_results = []
for tid in selected:
    track = gs.track(tid)
    print(f'  {tid} ({track.style})...', end=' ', flush=True)

    _, _, bp_events = predict(track.audio_mic_path)
    pred_intervals = np.array([[e[0], e[1]] for e in bp_events])
    pred_midi = np.array([float(e[2]) - 12 for e in bp_events])  # BP e[2] is MIDI

    gt = track.notes_all  # gt.pitches is MIDI
    scores = score_notes(gt.intervals, gt.pitches, pred_intervals, pred_midi)
    scores['track_id'] = tid
    scores['style'] = track.style
    bp_shift_results.append(scores)
    print(f'onset={scores["onset_f1"]:.3f}  pitch={scores["onset_pitch_f1"]:.3f}  '
          f'full={scores["full_f1"]:.3f}')


# ============================================
# Method 4: Re-score Basic Pitch with correct MIDI handling
# ============================================
print("\n" + "=" * 60)
print("METHOD 4: Basic Pitch (re-scored, correct MIDI)")
print("=" * 60)

bp_fixed_results = []
for tid in selected:
    track = gs.track(tid)
    print(f'  {tid} ({track.style})...', end=' ', flush=True)
    _, _, bp_events = predict(track.audio_mic_path)
    pred_intervals = np.array([[e[0], e[1]] for e in bp_events])
    pred_midi = np.array([float(e[2]) for e in bp_events])  # already MIDI
    gt = track.notes_all
    scores = score_notes(gt.intervals, gt.pitches, pred_intervals, pred_midi)
    scores['track_id'] = tid
    scores['style'] = track.style
    bp_fixed_results.append(scores)
    print(f'onset={scores["onset_f1"]:.3f}  pitch={scores["onset_pitch_f1"]:.3f}  '
          f'full={scores["full_f1"]:.3f}')


# ============================================
# COMPARISON TABLE
# ============================================
print("\n" + "=" * 60)
print("COMPARISON TABLE -- Average F1 across 5 solo tracks")
print("=" * 60)

methods = [
    ("Basic Pitch (corrected)", bp_fixed_results),
    ("Basic Pitch (-12 oct)", bp_shift_results),
    ("Our CQT Detector", our_results),
    ("Hybrid (BP onset+CQT)", hybrid_results),
]

print(f'{"Method":<30s} {"Onset F1":>10s} {"Pitch F1":>10s} {"Full F1":>10s}')
print("-" * 62)
for name, results in methods:
    if not results:
        print(f'{name:<30s} {"N/A":>10s} {"N/A":>10s} {"N/A":>10s}')
        continue
    avg1 = np.mean([r['onset_f1'] for r in results])
    avg2 = np.mean([r['onset_pitch_f1'] for r in results])
    avg3 = np.mean([r['full_f1'] for r in results])
    print(f'{name:<30s} {avg1:>10.3f} {avg2:>10.3f} {avg3:>10.3f}')

# Save all results
log = {
    'basic_pitch_corrected': bp_fixed_results,
    'basic_pitch_octave_shift': bp_shift_results,
    'our_cqt_detector': our_results,
    'hybrid_bp_onset_cqt_pitch': hybrid_results,
}
with open('output/evaluation_log.json', 'w') as f:
    json.dump(log, f, indent=2)
print('\nResults saved to output/evaluation_log.json')
