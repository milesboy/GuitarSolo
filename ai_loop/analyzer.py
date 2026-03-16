"""AI reasoning loop using Claude API.

Analyzes transcription failures against ground truth and suggests
targeted improvements. This is the novel part of the architecture —
Claude reasons about what went wrong and what to try next, rather
than brute-forcing parameters.
"""
import json
import os
import anthropic
import numpy as np
import mir_eval
import librosa


def build_analysis_prompt(track_id, style, tempo, gt_notes, pred_notes,
                          scores, error_details, note_comparison):
    """Build a detailed prompt for Claude to analyze transcription failures."""
    return f"""You are an expert guitar transcription engineer analyzing the output of an AI transcription system (Basic Pitch by Spotify) on a guitar recording.

## Track Info
- Track: {track_id}
- Style: {style}
- Tempo: {tempo} BPM
- Ground truth notes: {gt_notes}
- Predicted notes: {pred_notes}

## Scores (mir_eval, 50ms onset tolerance, 0.5 semitone pitch tolerance)
- Onset F1: {scores['onset_f1']:.3f} (P={scores['onset_p']:.3f}, R={scores['onset_r']:.3f})
- Onset+Pitch F1: {scores['pitch_f1']:.3f} (P={scores['pitch_p']:.3f}, R={scores['pitch_r']:.3f})
- Full F1 (onset+offset+pitch): {scores['full_f1']:.3f}

## Error Breakdown
{json.dumps(error_details, indent=2)}

## Note-by-Note Comparison (first 30 events)
{note_comparison}

## Your Task
Analyze the specific failure patterns and provide:

1. **Root causes**: What specific types of errors dominate? (octave errors, ghost notes, missed bass notes, duration issues, etc.)

2. **Actionable fixes**: For each root cause, suggest a specific post-processing step we could implement in Python. Be concrete — give threshold values, algorithms, or heuristics.

3. **Priority ranking**: Rank the fixes by expected impact (how many notes would each fix correct).

4. **Parameter suggestions**: If adjusting Basic Pitch parameters (onset_threshold, frame_threshold, minimum_note_length) would help, suggest specific values and why.

5. **Guitar-specific insights**: What guitar playing techniques might be confusing the detector? (harmonics, hammer-ons, open string ringing, etc.)

Format your response as JSON with keys: root_causes, fixes, priority, parameters, insights
"""


def compare_notes(gt_intervals, gt_midi, pred_intervals, pred_midi, max_notes=30):
    """Build a note-by-note comparison string for the prompt."""
    lines = []
    lines.append("Time     | GT Note | GT MIDI | Pred Note | Pred MIDI | Match?")
    lines.append("-" * 70)

    gt_times = gt_intervals[:, 0] if len(gt_intervals) > 0 else []
    pred_times = pred_intervals[:, 0] if len(pred_intervals) > 0 else []

    # Merge all event times
    all_times = sorted(set(
        list(gt_times[:max_notes]) + list(pred_times[:max_notes])
    ))[:max_notes]

    for t in all_times:
        # Find GT note near this time
        gt_match = None
        for i, (gs, ge) in enumerate(gt_intervals):
            if abs(gs - t) < 0.05:
                gt_match = (gs, gt_midi[i])
                break

        # Find pred note near this time
        pred_match = None
        for i, (ps, pe) in enumerate(pred_intervals):
            if abs(ps - t) < 0.05:
                pred_match = (ps, pred_midi[i])
                break

        gt_note = librosa.midi_to_note(int(gt_match[1])) if gt_match else "---"
        gt_m = f"{gt_match[1]:.0f}" if gt_match else "---"
        pred_note = librosa.midi_to_note(int(pred_match[1])) if pred_match else "---"
        pred_m = f"{pred_match[1]:.0f}" if pred_match else "---"

        match = ""
        if gt_match and pred_match:
            if abs(gt_match[1] - pred_match[1]) <= 0.5:
                match = "OK"
            elif abs(gt_match[1] - pred_match[1] - 12) <= 0.5 or \
                 abs(gt_match[1] - pred_match[1] + 12) <= 0.5:
                match = "OCTAVE"
            else:
                match = f"WRONG ({pred_match[1] - gt_match[1]:+.0f})"
        elif gt_match:
            match = "MISSED"
        else:
            match = "EXTRA"

        lines.append(f"{t:7.3f}s | {gt_note:>7s} | {gt_m:>7s} | {pred_note:>9s} | {pred_m:>9s} | {match}")

    return "\n".join(lines)


def analyze_errors(gt_intervals, gt_midi, pred_intervals, pred_midi):
    """Categorize errors for the AI prompt."""
    from collections import defaultdict
    errors = defaultdict(int)
    matched_ref = set()

    for pi in range(len(pred_intervals)):
        ps = pred_intervals[pi][0]
        pm = pred_midi[pi]
        best_dist = float('inf')
        best_ri = None

        for ri in range(len(gt_intervals)):
            if ri in matched_ref:
                continue
            if abs(ps - gt_intervals[ri][0]) <= 0.05:
                dist = abs(pm - gt_midi[ri])
                if dist < best_dist:
                    best_dist = dist
                    best_ri = ri

        if best_ri is not None and best_dist <= 0.5:
            matched_ref.add(best_ri)
            # Check duration
            gt_dur = gt_intervals[best_ri][1] - gt_intervals[best_ri][0]
            pred_dur = pred_intervals[pi][1] - pred_intervals[pi][0]
            if abs(gt_dur - pred_dur) / max(gt_dur, 0.01) > 0.5:
                errors['duration_error'] += 1
        elif best_ri is not None:
            diff = int(round(pm - gt_midi[best_ri]))
            if abs(diff) == 12:
                errors['octave_error'] += 1
            elif abs(diff) == 7:
                errors['fifth_error'] += 1
            elif abs(diff) <= 2:
                errors['semitone_error'] += 1
            else:
                errors['large_pitch_error'] += 1
            matched_ref.add(best_ri)
        else:
            errors['extra_note'] += 1

    errors['missed_note'] = len(gt_intervals) - len(matched_ref)
    errors['total_gt'] = len(gt_intervals)
    errors['total_pred'] = len(pred_intervals)

    return dict(errors)


def run_analysis(track, pred_notes, model="claude-sonnet-4-20250514"):
    """Run the AI analysis on a single track.

    Args:
        track: mirdata track object with .notes_all, .style, .tempo, etc.
        pred_notes: list of (time, note_name, freq, velocity, duration)
        model: Claude model to use

    Returns:
        dict with Claude's analysis
    """
    gt = track.notes_all
    gt_midi = np.round(np.array(gt.pitches, dtype=float))

    # Convert pred notes to intervals + midi
    pred_intervals = np.array([[n[0], n[0] + n[4]] for n in pred_notes])
    pred_midi = np.array([float(librosa.note_to_midi(
        n[1].replace("\u266f", "#").replace("\u266d", "b")))
        for n in pred_notes])

    # Score
    scores = {}
    for label, ptol, orat in [('onset', 50, None), ('pitch', 0.5, None), ('full', 0.5, 0.2)]:
        p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            gt.intervals, gt_midi, pred_intervals, pred_midi,
            onset_tolerance=0.05, pitch_tolerance=ptol,
            offset_ratio=orat)
        scores[f'{label}_f1'] = round(f1, 3)
        scores[f'{label}_p'] = round(p, 3)
        scores[f'{label}_r'] = round(r, 3)

    # Error details
    error_details = analyze_errors(gt.intervals, gt_midi, pred_intervals, pred_midi)

    # Note comparison
    note_comp = compare_notes(gt.intervals, gt_midi, pred_intervals, pred_midi)

    # Build prompt
    prompt = build_analysis_prompt(
        track_id=track.track_id if hasattr(track, 'track_id') else "unknown",
        style=track.style,
        tempo=track.tempo,
        gt_notes=len(gt.intervals),
        pred_notes=len(pred_notes),
        scores=scores,
        error_details=error_details,
        note_comparison=note_comp,
    )

    # Call Claude
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    # Parse response
    response_text = response.content[0].text

    # Try to extract JSON
    try:
        # Find JSON block in response
        import re
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            analysis = json.loads(json_match.group())
        else:
            analysis = {"raw_response": response_text}
    except json.JSONDecodeError:
        analysis = {"raw_response": response_text}

    analysis['scores'] = scores
    analysis['errors'] = error_details

    return analysis


def run_multi_track_analysis(tracks_and_preds, model="claude-sonnet-4-20250514",
                             verbose=True):
    """Analyze multiple tracks and synthesize findings.

    Args:
        tracks_and_preds: list of (track, pred_notes) tuples
        model: Claude model
        verbose: print progress

    Returns:
        list of per-track analyses + summary
    """
    analyses = []

    for i, (track, pred_notes) in enumerate(tracks_and_preds):
        if verbose:
            print(f"  [{i+1}/{len(tracks_and_preds)}] Analyzing {track.style}...",
                  end=" ", flush=True)
        try:
            analysis = run_analysis(track, pred_notes, model=model)
            analyses.append(analysis)
            if verbose:
                print(f"pitch_f1={analysis['scores']['pitch_f1']:.3f}")
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            analyses.append({"error": str(e)})

    return analyses
