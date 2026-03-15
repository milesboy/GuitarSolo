"""Note-level refinement pass.

After initial detection, walks through the audio at 64th-note resolution
to find missed notes, verify pitch accuracy, and correct durations.

Key principles:
- New notes require a clear PLUCK SIGNATURE (energy spike vs pre-onset
  level) — natural decay of a ringing note is NOT a new note
- Notes above E5 (bin 36) are filtered as likely harmonics/artifacts
- Song sections are processed in parallel for speed
"""
import librosa
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Maximum CQT bin for real guitar notes (E5 = bin 36)
# Notes above this are almost certainly harmonics, not fretted notes
MAX_GUITAR_BIN = 36

# Minimum energy spike ratio to confirm a pluck (onset vs pre-onset)
PLUCK_RATIO = 2.0


def _build_cqt(y, sr, hop_length=256):
    """Build a high-resolution CQT for note verification."""
    y_harmonic, _ = librosa.effects.hpss(y)
    tuning_offset = librosa.estimate_tuning(y=y_harmonic, sr=sr)
    fmin = librosa.note_to_hz("E2") * (2 ** (tuning_offset / 12))

    n_semitones = 48
    bins_per_semi = 3
    cqt_raw = np.abs(librosa.cqt(
        y=y_harmonic, sr=sr, fmin=fmin, hop_length=hop_length,
        n_bins=n_semitones * bins_per_semi,
        bins_per_octave=12 * bins_per_semi,
    ))

    cqt_semitone = np.zeros((n_semitones, cqt_raw.shape[1]))
    for i in range(n_semitones):
        cqt_semitone[i] = np.max(
            cqt_raw[i * bins_per_semi:(i + 1) * bins_per_semi], axis=0)

    fmin_standard = librosa.note_to_hz("E2")
    bin_notes = []
    bin_freqs = []
    for i in range(n_semitones):
        midi_note = librosa.hz_to_midi(fmin_standard) + i
        bin_notes.append(librosa.midi_to_note(midi_note))
        bin_freqs.append(float(librosa.midi_to_hz(midi_note)))

    return cqt_semitone, bin_notes, bin_freqs, hop_length, y_harmonic


def _has_pluck_signature(cqt, note_bin, onset_frame, hop_length, sr):
    """Check if there's a genuine pluck transient at this onset.

    Compares CQT energy at the onset frame against a window just before.
    A real pluck shows a sharp energy increase; natural decay does not.

    Returns True if energy at onset is >= PLUCK_RATIO times pre-onset energy.
    """
    # Look 3-5 frames before onset (~30-50ms at hop=256)
    pre_frames = 4
    pre_start = max(0, onset_frame - pre_frames)

    if onset_frame <= pre_start:
        return True  # at the very start — can't check, assume pluck

    pre_energy = float(np.mean(cqt[note_bin, pre_start:onset_frame]))
    window = min(3, cqt.shape[1] - onset_frame)
    if window <= 0:
        return False
    onset_energy = float(np.mean(cqt[note_bin, onset_frame:onset_frame + window]))

    if pre_energy <= 0:
        return onset_energy > 0  # silence -> sound = pluck

    return onset_energy / pre_energy >= PLUCK_RATIO


def _find_harmonics(cqt, bin_notes, bin_freqs, notes, bpm, sr,
                    hop_length=256, verbose=True):
    """Scan for natural harmonics (gradual onset, no sharp transient).

    Natural harmonics emerge when a guitarist lightly touches the string
    at a node point. Unlike plucked notes, they build up gradually over
    ~200-400ms rather than having a sharp attack. They also produce a
    very pure tone at specific harmonic ratios of the open string.

    Scans at 64th-note intervals looking for bins where energy rises
    significantly over a ~300ms window without a conventional onset.
    """
    sec_per_64th = 60.0 / bpm / 16
    n_semitones = cqt.shape[0]
    window = max(1, int(0.15 * sr / hop_length))
    rise_window = max(1, int(0.35 * sr / hop_length))  # ~350ms to detect gradual rise

    # Open string MIDI notes and their harmonic frequencies
    # 5th fret harmonic = 4th harmonic (2 octaves up)
    # 7th fret harmonic = 3rd harmonic (octave + 5th)
    # 12th fret harmonic = 2nd harmonic (octave)
    open_strings_midi = [40, 45, 50, 55, 59, 64]  # E2 A2 D3 G3 B3 E4
    harmonic_bins = set()
    for open_midi in open_strings_midi:
        for multiplier, fret_name in [(2, "12th"), (3, "7th"), (4, "5th")]:
            harm_midi = open_midi + round(12 * np.log2(multiplier))
            if 0 <= harm_midi - 40 < n_semitones:
                harmonic_bins.add(harm_midi - 40)  # bin index

    # Build set of times already covered by notes
    occupied_onsets = set()
    for t, note, freq, vel, dur in notes:
        slot = round(t / sec_per_64th)
        for s in range(slot - 2, slot + 3):
            occupied_onsets.add(s)

    all_mags = cqt[cqt > 0]
    if len(all_mags) == 0:
        return []
    threshold = np.percentile(all_mags, 65)

    # Scan at every 8th-note position (every 8 64th-note slots)
    total_frames = cqt.shape[1]
    max_time = librosa.frames_to_time(total_frames, sr=sr, hop_length=hop_length)
    scan_step = sec_per_64th * 8  # check every 8th note

    new_harmonics = []
    t = 0.5  # skip the very start

    while t < max_time - 0.5:
        slot = round(t / sec_per_64th)
        if slot in occupied_onsets:
            t += scan_step
            continue

        frame = librosa.time_to_frames(t, sr=sr, hop_length=hop_length)
        if frame >= total_frames or frame < rise_window:
            t += scan_step
            continue

        # Check energy at this point vs ~350ms before
        for b in harmonic_bins:
            if b >= MAX_GUITAR_BIN:
                continue

            now_energy = float(np.mean(
                cqt[b, frame:min(frame + window, total_frames)]))
            before_energy = float(np.mean(
                cqt[b, frame - rise_window:frame]))

            if now_energy < threshold:
                continue

            # Harmonic detection: at a known harmonic frequency, any
            # growth (10%+) means new energy is being added — natural
            # decay would show a declining ratio.  Require significant
            # absolute energy to reject noise.
            if before_energy > 0 and now_energy > threshold * 1.5:
                ratio = now_energy / before_energy
                if 1.1 <= ratio < PLUCK_RATIO:
                    # Confirm it's a local peak (not leakage)
                    mag_slice = np.mean(
                        cqt[:, frame:min(frame + window, total_frames)], axis=1)
                    left = mag_slice[b - 1] if b > 0 else 0
                    right = mag_slice[b + 1] if b < n_semitones - 1 else 0
                    if not (mag_slice[b] > left and mag_slice[b] > right):
                        continue

                    note_name = bin_notes[b]
                    freq = bin_freqs[b]
                    vel = int(min(100, max(40, 40 + 60 * (now_energy / threshold) ** 0.35)))

                    # Duration: scan forward
                    dur = 0.2
                    check_frame = frame + int(0.1 * sr / hop_length)
                    while check_frame < total_frames:
                        e = float(np.mean(cqt[b, check_frame:min(
                            check_frame + window, total_frames)]))
                        if e < now_energy * 0.2:
                            break
                        dur += 0.1
                        check_frame += int(0.1 * sr / hop_length)
                        if dur > 4.0:
                            break

                    new_harmonics.append((t, note_name, freq, vel, dur))
                    # Mark occupied
                    for s in range(slot - 2, slot + int(dur / sec_per_64th) + 2):
                        occupied_onsets.add(s)
                    break  # one harmonic per time position

        t += scan_step

    if verbose and new_harmonics:
        print(f"  Found {len(new_harmonics)} harmonics:")
        for ht, hn, hf, hv, hd in new_harmonics[:5]:
            print(f"    t={ht:.3f}s {hn} ({hf:.1f}Hz) dur={hd:.2f}s")

    return new_harmonics


def find_missing_notes(y, sr, notes, bpm, cqt=None, bin_notes=None,
                       bin_freqs=None, y_harm=None, verbose=True):
    """Scan for notes that the detector missed.

    Two passes:
    1. Pluck scan: conventional onsets with sharp transients
    2. Harmonic scan: gradual energy rises at harmonic frequencies

    Filters out notes above E5 (likely artifacts).

    Args:
        y: audio signal
        sr: sample rate
        notes: list of (time, note_name, freq, velocity, duration)
        bpm: tempo
        cqt, bin_notes, bin_freqs, y_harm: pre-built CQT (optional)

    Returns:
        list of new notes to add (same format as input notes)
    """
    hop_length = 256
    if cqt is None:
        cqt, bin_notes, bin_freqs, _, y_harm = _build_cqt(y, sr, hop_length)

    n_semitones = cqt.shape[0]
    sec_per_64th = 60.0 / bpm / 16

    # Build occupied onset slots (±1 slot around each existing note)
    occupied_onsets = set()
    for t, note, freq, vel, dur in notes:
        onset_slot = round(t / sec_per_64th)
        for s in range(onset_slot - 1, onset_slot + 2):
            occupied_onsets.add(s)

    # Global energy threshold
    all_mags = cqt[cqt > 0]
    if len(all_mags) == 0:
        return []
    threshold = np.percentile(all_mags, 60)

    # Onset detection
    onset_frames = librosa.onset.onset_detect(
        y=y_harm, sr=sr, hop_length=hop_length, backtrack=True)
    onset_times = librosa.frames_to_time(
        onset_frames, sr=sr, hop_length=hop_length)

    window_frames = max(1, int(0.15 * sr / hop_length))

    new_notes = []

    for onset_t in onset_times:
        slot = int(onset_t / sec_per_64th)
        if slot in occupied_onsets:
            continue

        onset_frame = librosa.time_to_frames(
            onset_t, sr=sr, hop_length=hop_length)
        start = onset_frame
        end = min(start + window_frames, cqt.shape[1])
        if start >= cqt.shape[1]:
            continue

        mag_window = np.mean(cqt[:, start:end], axis=1)
        max_mag = np.max(mag_window)
        if max_mag < threshold:
            continue

        # Find strongest peak WITHIN guitar range (below E5)
        guitar_mags = mag_window[:MAX_GUITAR_BIN].copy()
        if np.max(guitar_mags) < threshold:
            continue
        best_bin = np.argmax(guitar_mags)

        # Verify it's a local spectral peak
        left = mag_window[best_bin - 1] if best_bin > 0 else 0
        right = mag_window[best_bin + 1] if best_bin < n_semitones - 1 else 0
        if not (mag_window[best_bin] > left and mag_window[best_bin] > right):
            continue

        # CRITICAL: verify pluck signature — must be a new string attack,
        # not just the natural decay of a ringing note
        if not _has_pluck_signature(cqt, best_bin, onset_frame, hop_length, sr):
            continue

        note_name = bin_notes[best_bin]
        freq = bin_freqs[best_bin]
        mag = float(mag_window[best_bin])

        vel = int(min(127, max(50, 50 + 77 * (mag / max_mag) ** 0.35)))

        # Estimate duration
        check_interval = int(0.1 * sr / hop_length)
        dur = 0.15
        check_frame = start + check_interval
        while check_frame < cqt.shape[1]:
            c_end = min(check_frame + window_frames, cqt.shape[1])
            energy = np.mean(cqt[best_bin, check_frame:c_end])
            if energy < mag * 0.2:
                break
            dur += 0.1
            check_frame += check_interval
            if dur > 4.0:
                break

        new_notes.append((onset_t, note_name, freq, vel, dur))
        for s in range(slot - 1, slot + 2):
            occupied_onsets.add(s)

    if verbose and new_notes:
        print(f"  Found {len(new_notes)} missing plucked notes")

    # Pass 2: scan for natural harmonics (gradual onset)
    all_notes_so_far = list(notes) + new_notes
    harmonics = _find_harmonics(
        cqt, bin_notes, bin_freqs, all_notes_so_far, bpm, sr,
        hop_length, verbose=verbose)

    return new_notes + harmonics


def _verify_chunk(args):
    """Verify a chunk of notes (for parallel processing)."""
    chunk, cqt, bin_notes, bin_freqs, sr, hop_length, n_semitones = args
    window_frames = max(1, int(0.15 * sr / hop_length))

    corrected = []
    corrections = []

    for i, (t, note_name, freq, vel, dur) in chunk:
        onset_frame = librosa.time_to_frames(t, sr=sr, hop_length=hop_length)
        start = onset_frame
        end = min(start + window_frames, cqt.shape[1])
        if start >= cqt.shape[1]:
            corrected.append((i, (t, note_name, freq, vel, dur)))
            continue

        mag_window = np.mean(cqt[:, start:end], axis=1)

        # --- Pitch verification ---
        if note_name in bin_notes:
            expected_bin = bin_notes.index(note_name)
        else:
            corrected.append((i, (t, note_name, freq, vel, dur)))
            continue

        # Filter out notes above guitar range
        if expected_bin >= MAX_GUITAR_BIN:
            # Check if there's a stronger fundamental below
            guitar_mags = mag_window[:MAX_GUITAR_BIN]
            if np.max(guitar_mags) > mag_window[expected_bin] * 0.5:
                best_bin = np.argmax(guitar_mags)
                note_name = bin_notes[best_bin]
                freq = bin_freqs[best_bin]
                corrections.append((i, "pitch", bin_notes[expected_bin],
                                    f"{note_name} (was above range)"))
                expected_bin = best_bin
            else:
                corrected.append((i, (t, note_name, freq, vel, dur)))
                continue

        # Check ±1 semitone for a stronger peak
        best_bin = expected_bin
        best_mag = mag_window[expected_bin]
        for offset in [-1, 1]:
            check_bin = expected_bin + offset
            if 0 <= check_bin < min(n_semitones, MAX_GUITAR_BIN):
                if mag_window[check_bin] > best_mag * 1.3:
                    best_bin = check_bin
                    best_mag = mag_window[check_bin]

        if best_bin != expected_bin:
            new_note = bin_notes[best_bin]
            new_freq = bin_freqs[best_bin]
            corrections.append((i, "pitch", note_name, new_note))
            note_name = new_note
            freq = new_freq

        # --- Duration verification ---
        check_interval = int(0.1 * sr / hop_length)
        onset_mag = mag_window[best_bin]
        actual_dur = 0.15
        check_frame = start + check_interval
        while check_frame < cqt.shape[1]:
            c_end = min(check_frame + window_frames, cqt.shape[1])
            energy = np.mean(cqt[best_bin, check_frame:c_end])
            if energy < onset_mag * 0.2:
                break
            actual_dur += 0.1
            check_frame += check_interval
            if actual_dur > 4.0:
                break

        if abs(actual_dur - dur) / max(dur, 0.01) > 0.3:
            corrections.append((i, "duration", f"{dur:.2f}",
                                f"{actual_dur:.2f}"))
            dur = actual_dur

        corrected.append((i, (t, note_name, freq, vel, dur)))

    return corrected, corrections


def verify_notes(y, sr, notes, cqt=None, bin_notes=None, bin_freqs=None,
                 verbose=True):
    """Verify pitch and duration of each detected note against the CQT.

    Processes song sections in parallel for speed.
    Filters notes above E5 (bin 36) as likely harmonics.
    """
    hop_length = 256
    if cqt is None:
        cqt, bin_notes, bin_freqs, _, _ = _build_cqt(y, sr, hop_length)
    n_semitones = cqt.shape[0]

    # Split notes into chunks for parallel processing
    indexed_notes = list(enumerate(notes))
    n_workers = 4
    chunk_size = max(1, len(indexed_notes) // n_workers)
    chunks = []
    for start in range(0, len(indexed_notes), chunk_size):
        chunk = indexed_notes[start:start + chunk_size]
        chunks.append((chunk, cqt, bin_notes, bin_freqs,
                        sr, hop_length, n_semitones))

    # Process chunks in parallel using threads (shared CQT memory)
    all_corrected = []
    all_corrections = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_verify_chunk, chunks))

    for corrected_chunk, corrections_chunk in results:
        all_corrected.extend(corrected_chunk)
        all_corrections.extend(corrections_chunk)

    # Sort by original index to maintain order
    all_corrected.sort(key=lambda x: x[0])
    corrected_notes = [note for _, note in all_corrected]

    if verbose:
        print(f"  Verified {len(notes)} notes ({n_workers} threads), "
              f"{len(all_corrections)} corrections")
        for idx, field, old, new in all_corrections[:10]:
            print(f"    note {idx}: {field} {old} -> {new}")
        if len(all_corrections) > 10:
            print(f"    ... and {len(all_corrections) - 10} more")

    return corrected_notes, all_corrections


def refine_notes(y, sr, notes, bpm, verbose=True):
    """Full refinement pass: verify existing notes, then find missing ones.

    - Verifies pitch and duration of each note against CQT
    - Filters out notes above guitar range (likely harmonics)
    - Scans for missed notes with confirmed pluck signatures
    - Processes sections in parallel

    Returns:
        refined_notes: corrected + newly found notes, sorted by time
    """
    if verbose:
        print("\n=== Note Refinement Pass ===")
        print(f"  Input: {len(notes)} notes")

    # Build CQT once, share across verify and find_missing
    hop_length = 256
    cqt, bin_notes, bin_freqs, _, y_harm = _build_cqt(y, sr, hop_length)

    # Step 1: verify pitch and duration
    verified, corrections = verify_notes(
        y, sr, notes, cqt=cqt, bin_notes=bin_notes,
        bin_freqs=bin_freqs, verbose=verbose)

    # Step 2: find missing notes (pluck signature required)
    missing = find_missing_notes(
        y, sr, verified, bpm, cqt=cqt, bin_notes=bin_notes,
        bin_freqs=bin_freqs, y_harm=y_harm, verbose=verbose)

    # Merge and sort
    refined = sorted(verified + missing, key=lambda x: x[0])

    if verbose:
        print(f"  Output: {len(refined)} notes "
              f"({len(corrections)} corrected, {len(missing)} added)")

    return refined
