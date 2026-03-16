"""Spectral difference analysis: render MIDI to WAV, compare against original.

Finds extra notes (energy in rendered not in original) and missing notes
(energy in original not in rendered) by comparing CQT spectrograms.
"""
import subprocess
import os
import tempfile
import numpy as np
import librosa

from config import FLUIDSYNTH_PATH, SOUNDFONT_PATH


def render_midi_to_wav(midi_path, wav_path=None):
    """Render MIDI to WAV using FluidSynth (melody track only).

    Strips the chords track before rendering so only the transcription
    is compared against the original audio.
    """
    import pretty_midi

    if not os.path.isfile(FLUIDSYNTH_PATH):
        raise RuntimeError(f"FluidSynth not found at {FLUIDSYNTH_PATH}")
    if not os.path.isfile(SOUNDFONT_PATH):
        raise RuntimeError(f"Soundfont not found at {SOUNDFONT_PATH}")

    if wav_path is None:
        wav_path = midi_path.replace('.mid', '_rendered.wav')

    # Write a melody-only MIDI with drawbar organ (fewer harmonics
    # than guitar = cleaner spectral comparison)
    ORGAN_PROGRAM = 16  # GM Drawbar Organ
    mid = pretty_midi.PrettyMIDI(midi_path)
    melody_only = pretty_midi.PrettyMIDI(initial_tempo=mid.estimate_tempo())
    for inst in mid.instruments:
        if inst.name == "Melody" or len(mid.instruments) == 1:
            inst.program = ORGAN_PROGRAM  # switch to organ
            melody_only.instruments.append(inst)
            break
    if not melody_only.instruments and mid.instruments:
        mid.instruments[0].program = ORGAN_PROGRAM
        melody_only.instruments.append(mid.instruments[0])

    tmp_midi = midi_path.replace('.mid', '_melody_only.mid')
    melody_only.write(tmp_midi)

    cmd = [FLUIDSYNTH_PATH, "-a", "file", "-F", wav_path,
           "-r", "22050", "-ni", SOUNDFONT_PATH, tmp_midi]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    # Clean up temp file
    try:
        os.unlink(tmp_midi)
    except Exception:
        pass

    if result.returncode != 0:
        raise RuntimeError(f"FluidSynth failed: {result.stderr}")

    return wav_path


def compute_cqt_db(y, sr, hop_length=512, n_semitones=52):
    """Compute dB-normalized CQT."""
    y_harm, _ = librosa.effects.hpss(y)
    tuning = librosa.estimate_tuning(y=y_harm, sr=sr)
    fmin = librosa.note_to_hz("E2") * (2 ** (tuning / 12))

    bins_per_semi = 3
    cqt_raw = np.abs(librosa.cqt(
        y=y_harm, sr=sr, fmin=fmin, hop_length=hop_length,
        n_bins=n_semitones * bins_per_semi,
        bins_per_octave=12 * bins_per_semi))

    cqt = np.zeros((n_semitones, cqt_raw.shape[1]))
    for i in range(n_semitones):
        cqt[i] = np.max(cqt_raw[i*bins_per_semi:(i+1)*bins_per_semi], axis=0)

    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max(cqt) if np.max(cqt) > 0 else 1.0)
    return cqt_db, hop_length


def find_spectral_differences(original_wav, midi_path, verbose=True):
    """Compare original audio against rendered MIDI to find errors.

    Returns:
        dict with:
        - extra_notes: list of (time, midi, magnitude) — in MIDI but not original
        - missing_notes: list of (time, midi, magnitude) — in original but not MIDI
        - overall_similarity: 0-1 score
    """
    if verbose:
        print("  Rendering MIDI to WAV...", end=" ", flush=True)
    rendered_wav = render_midi_to_wav(midi_path)
    if verbose:
        print("done")

    # Load both at same sample rate
    sr = 22050
    y_orig, _ = librosa.load(original_wav, sr=sr)
    y_rend, _ = librosa.load(rendered_wav, sr=sr)

    # Match lengths
    min_len = min(len(y_orig), len(y_rend))
    y_orig = y_orig[:min_len]
    y_rend = y_rend[:min_len]

    if verbose:
        print("  Computing spectrograms...", end=" ", flush=True)

    hop = 512
    n_semi = 52
    orig_db, _ = compute_cqt_db(y_orig, sr, hop, n_semi)
    rend_db, _ = compute_cqt_db(y_rend, sr, hop, n_semi)

    # Match frame counts
    min_frames = min(orig_db.shape[1], rend_db.shape[1])
    orig_db = orig_db[:, :min_frames]
    rend_db = rend_db[:, :min_frames]

    if verbose:
        print("done")

    # Compute difference: positive = extra in rendered, negative = missing
    diff = rend_db - orig_db

    # Frequency-weighted thresholds — bass has more ambient energy,
    # treble extras are more audible and easier to confirm
    BASS_BIN_MAX = 15  # bins 0-15 = E2 to G3 (bass range)
    EXTRA_THRESH_BASS = 55.0    # higher bar for bass extras
    EXTRA_THRESH_TREBLE = 35.0  # lower bar for treble extras
    MISSING_THRESHOLD = 40.0

    fmin_std = librosa.note_to_hz("E2")
    bin_midi = [int(round(librosa.hz_to_midi(fmin_std) + i)) for i in range(n_semi)]
    bin_names = [librosa.midi_to_note(m) for m in bin_midi]

    # Find extra notes — only report if energy is strong in rendered
    # AND weak in original (not just FluidSynth timbre harmonics)
    extra_notes = []
    window = max(1, int(0.25 * sr / hop))
    for frame in range(0, min_frames, window):
        end = min(frame + window, min_frames)
        for b in range(n_semi):
            thresh = EXTRA_THRESH_BASS if b <= BASS_BIN_MAX else EXTRA_THRESH_TREBLE
            if np.any(diff[b, frame:end] > thresh):
                # Extra check: original must be quiet at this bin
                # (otherwise it's timbre difference, not a wrong note)
                orig_energy = float(np.max(orig_db[b, frame:end]))
                if orig_energy > -30:
                    continue  # original has energy here too — just timbre
                mag = float(np.max(diff[b, frame:end]))
                t = librosa.frames_to_time(frame, sr=sr, hop_length=hop)
                extra_notes.append({
                    'time': round(float(t), 3),
                    'midi': bin_midi[b],
                    'note': bin_names[b],
                    'excess_db': round(mag, 1),
                })

    # Find missing notes with frequency-weighted thresholds
    MISSING_THRESH_BASS = 55.0   # higher bar for bass missing
    MISSING_THRESH_TREBLE = 35.0 # lower bar for treble missing
    missing_notes = []
    for frame in range(0, min_frames, window):
        end = min(frame + window, min_frames)
        for b in range(n_semi):
            thresh = MISSING_THRESH_BASS if b <= BASS_BIN_MAX else MISSING_THRESH_TREBLE
            if np.any(diff[b, frame:end] < -thresh):
                mag = float(np.min(diff[b, frame:end]))
                t = librosa.frames_to_time(frame, sr=sr, hop_length=hop)
                missing_notes.append({
                    'time': round(float(t), 3),
                    'midi': bin_midi[b],
                    'note': bin_names[b],
                    'deficit_db': round(abs(mag), 1),
                })

    # Deduplicate (same note within 500ms)
    def dedup(notes_list, key='note'):
        seen = {}
        deduped = []
        for n in notes_list:
            k = (n[key], round(n['time'] / 0.5))
            if k not in seen:
                seen[k] = True
                deduped.append(n)
        return deduped

    extra_notes = dedup(extra_notes)
    missing_notes = dedup(missing_notes)

    # Overall similarity (chroma correlation)
    orig_chroma = np.mean(librosa.feature.chroma_cqt(y=y_orig, sr=sr), axis=1)
    rend_chroma = np.mean(librosa.feature.chroma_cqt(y=y_rend, sr=sr), axis=1)
    norm_o = np.linalg.norm(orig_chroma)
    norm_r = np.linalg.norm(rend_chroma)
    if norm_o > 0 and norm_r > 0:
        similarity = float(np.dot(orig_chroma, rend_chroma) / (norm_o * norm_r))
    else:
        similarity = 0.0

    # Clean up temp file
    try:
        os.unlink(rendered_wav)
    except Exception:
        pass

    if verbose:
        print(f"  Spectral comparison:")
        print(f"    Extra notes (in MIDI, not in original): {len(extra_notes)}")
        print(f"    Missing notes (in original, not in MIDI): {len(missing_notes)}")
        print(f"    Chroma similarity: {similarity:.3f}")

        if extra_notes:
            print(f"    Top extra notes:")
            for n in sorted(extra_notes, key=lambda x: -x['excess_db'])[:5]:
                print(f"      t={n['time']:.1f}s {n['note']} (+{n['excess_db']}dB)")

        if missing_notes:
            print(f"    Top missing notes:")
            for n in sorted(missing_notes, key=lambda x: -x['deficit_db'])[:5]:
                print(f"      t={n['time']:.1f}s {n['note']} (-{n['deficit_db']}dB)")

    return {
        'extra_notes': extra_notes,
        'missing_notes': missing_notes,
        'similarity': round(similarity, 3),
        'n_extra': len(extra_notes),
        'n_missing': len(missing_notes),
    }
