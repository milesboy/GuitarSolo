"""Run spectral difference analysis on a transcribed song.

Renders our MIDI back to WAV, compares against the original audio,
and reports extra notes and missing notes.

Usage:
    python run_spectral_diff.py downloads/song.wav
    python run_spectral_diff.py  # interactive picker
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scoring.spectral_diff import find_spectral_differences


def main():
    if len(sys.argv) > 1:
        wav_path = sys.argv[1]
    else:
        from extract import pick_song
        wav_path = pick_song()

    midi_path = os.path.splitext(wav_path)[0] + ".mid"
    if not os.path.exists(midi_path):
        print(f"No MIDI found at {midi_path}")
        print("Run the pipeline first: python pipeline.py " + wav_path)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Spectral Difference Analysis")
    print(f"  Original: {os.path.basename(wav_path)}")
    print(f"  MIDI:     {os.path.basename(midi_path)}")
    print(f"{'='*60}\n")

    results = find_spectral_differences(wav_path, midi_path, verbose=True)

    print(f"\n{'='*60}")
    print(f"  Summary:")
    print(f"  Chroma similarity: {results['similarity']:.1%}")
    print(f"  Extra notes:       {results['n_extra']}")
    print(f"  Missing notes:     {results['n_missing']}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
