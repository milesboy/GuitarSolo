"""Run AI analysis loop on GuitarSet tracks.

Sends transcription failures to Claude for analysis and gets back
specific, actionable recommendations for improvement.
"""
import sys; sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')
import json
import os
import mirdata
import librosa
from basic_pitch.inference import predict
from optimization.bp_postprocess import postprocess_bp
from ai_loop.analyzer import run_analysis


def main():
    gs = mirdata.initialize('guitarset', data_home='./data/guitarset')

    # Pick 5 diverse tracks (mix of solo/comp, different styles)
    targets = [
        ('solo', 'Bossa Nova'),
        ('solo', 'Rock'),
        ('comp', 'Jazz'),
        ('comp', 'Funk'),
        ('solo', 'Singer-Songwriter'),
    ]

    selected = []
    for mode, style in targets:
        for tid in gs.track_ids:
            t = gs.track(tid)
            if t.mode == mode and t.style == style:
                selected.append(tid)
                break

    print(f"Running AI analysis on {len(selected)} tracks...")
    print(f"{'='*60}")

    all_analyses = []

    for tid in selected:
        track = gs.track(tid)
        print(f"\n--- {tid} ({track.style}, {track.mode}, {track.tempo}bpm) ---")

        # Run Basic Pitch
        print("  Running Basic Pitch...", end=" ", flush=True)
        _, _, events = predict(track.audio_mic_path)
        bp_notes = [(float(e[0]), librosa.midi_to_note(int(e[2])),
                     float(librosa.midi_to_hz(int(e[2]))), int(e[3]*127),
                     float(e[1]-e[0])) for e in events]
        print(f"{len(bp_notes)} raw notes")

        # Post-process
        print("  Post-processing...", end=" ", flush=True)
        pp_notes = postprocess_bp(bp_notes, verbose=False)
        print(f"{len(pp_notes)} after PP")

        # Run AI analysis
        print("  Calling Claude for analysis...", end=" ", flush=True)
        try:
            analysis = run_analysis(track, pp_notes)
            all_analyses.append({
                'track_id': tid,
                'style': track.style,
                'mode': track.mode,
                'tempo': track.tempo,
                'analysis': analysis,
            })
            print("done")

            # Print key findings
            if 'root_causes' in analysis:
                print("  Root causes:")
                if isinstance(analysis['root_causes'], list):
                    for cause in analysis['root_causes'][:3]:
                        if isinstance(cause, dict):
                            print(f"    - {cause.get('cause', cause)}")
                        else:
                            print(f"    - {cause}")
                elif isinstance(analysis['root_causes'], str):
                    print(f"    {analysis['root_causes'][:200]}")

            if 'priority' in analysis:
                print("  Priority fixes:")
                if isinstance(analysis['priority'], list):
                    for fix in analysis['priority'][:3]:
                        if isinstance(fix, dict):
                            print(f"    {fix.get('rank', '?')}. {fix.get('fix', fix.get('description', fix))}")
                        else:
                            print(f"    - {fix}")

            print(f"  Scores: pitch_f1={analysis.get('scores', {}).get('pitch_f1', '?')} "
                  f"full_f1={analysis.get('scores', {}).get('full_f1', '?')}")

        except Exception as e:
            print(f"ERROR: {e}")
            all_analyses.append({
                'track_id': tid,
                'error': str(e),
            })

    # Save all results
    os.makedirs('output', exist_ok=True)
    with open('output/ai_analysis.json', 'w') as f:
        json.dump(all_analyses, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Full analysis saved to output/ai_analysis.json")

    # Print synthesis
    print(f"\n{'='*60}")
    print("SYNTHESIS — Common patterns across all tracks:")
    print(f"{'='*60}")

    valid = [a for a in all_analyses if 'error' not in a]
    if valid:
        # Collect all root causes
        all_causes = []
        all_fixes = []
        for a in valid:
            analysis = a['analysis']
            if 'root_causes' in analysis:
                rc = analysis['root_causes']
                if isinstance(rc, list):
                    all_causes.extend(rc)
            if 'fixes' in analysis:
                fx = analysis['fixes']
                if isinstance(fx, list):
                    all_fixes.extend(fx)

        print(f"\nAnalyzed {len(valid)} tracks successfully.")
        print(f"Total root causes identified: {len(all_causes)}")
        print(f"Total fixes suggested: {len(all_fixes)}")
        print("\nSee output/ai_analysis.json for full details.")


if __name__ == '__main__':
    main()
