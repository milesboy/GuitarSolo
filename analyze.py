import librosa
import numpy as np
from midiutil import MIDIFile
import sys
import os

# Map chroma indices to note names
CHROMA_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Major and minor key profiles (Krumhansl-Kessler)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def detect_tempo(y, sr):
    """Detect BPM."""
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(np.round(tempo[0] if hasattr(tempo, '__len__') else tempo))


def detect_key(y, sr):
    """Detect musical key using chroma features and key profiles."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)

    best_corr = -1
    best_key = "C major"

    for i in range(12):
        rotated = np.roll(chroma_avg, -i)
        major_corr = np.corrcoef(rotated, MAJOR_PROFILE)[0, 1]
        minor_corr = np.corrcoef(rotated, MINOR_PROFILE)[0, 1]

        if major_corr > best_corr:
            best_corr = major_corr
            best_key = f"{CHROMA_NOTES[i]} major"
        if minor_corr > best_corr:
            best_corr = minor_corr
            best_key = f"{CHROMA_NOTES[i]} minor"

    return best_key


def detect_chords(y, sr, hop_length=512):
    """Simple chord detection using chroma features."""
    # Common chord templates (root position triads)
    CHORD_TEMPLATES = {}
    for i, note in enumerate(CHROMA_NOTES):
        # Major triad: root, major third (+4), fifth (+7)
        major = np.zeros(12)
        major[i] = 1
        major[(i + 4) % 12] = 1
        major[(i + 7) % 12] = 1
        CHORD_TEMPLATES[note] = major

        # Minor triad: root, minor third (+3), fifth (+7)
        minor = np.zeros(12)
        minor[i] = 1
        minor[(i + 3) % 12] = 1
        minor[(i + 7) % 12] = 1
        CHORD_TEMPLATES[f"{note}m"] = minor

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(range(chroma.shape[1]), sr=sr, hop_length=hop_length)

    # Analyze in ~2-second windows for stability
    window_sec = 2.0
    window_frames = int(window_sec * sr / hop_length)

    chords = []
    for start in range(0, chroma.shape[1], window_frames):
        end = min(start + window_frames, chroma.shape[1])
        segment = np.mean(chroma[:, start:end], axis=1)
        segment = segment / (np.linalg.norm(segment) + 1e-6)

        best_chord = "N/C"
        best_score = -1

        for name, template in CHORD_TEMPLATES.items():
            norm_template = template / (np.linalg.norm(template) + 1e-6)
            score = np.dot(segment, norm_template)
            if score > best_score:
                best_score = score
                best_chord = name

        time_stamp = times[start] if start < len(times) else times[-1]
        chords.append((time_stamp, best_chord))

    # Collapse consecutive duplicates
    collapsed = [chords[0]]
    for time_stamp, chord in chords[1:]:
        if chord != collapsed[-1][1]:
            collapsed.append((time_stamp, chord))

    return collapsed


def detect_notes(y, sr, bpm=120.0):
    """Detect notes using CQT-based polyphonic pitch detection.

    Key technique: local-peak detection in CQT magnitude eliminates spectral
    leakage (phantom notes adjacent to real ones).  The lowest peak at each
    onset is treated as the bass; peaks that are NOT perfect harmonics of the
    bass (±1 semitone tolerance) are kept as real melody / harmony notes.
    Each confirmed real note then suppresses ITS harmonics from the remaining
    candidates.  Between onsets, sustained energy is detected so held notes
    (like a ringing G#4) are not dropped.
    """
    # Separate harmonic content from percussive
    y_harmonic, _ = librosa.effects.hpss(y)

    # Detect tuning offset and adjust fmin so CQT bins align to actual pitches
    tuning_offset = librosa.estimate_tuning(y=y_harmonic, sr=sr)
    fmin = librosa.note_to_hz("E2") * (2 ** (tuning_offset / 12))

    # High-resolution CQT: 3 bins per semitone, then collapse to nearest semitone
    hop_length = 512
    n_semitones = 48
    bins_per_semi = 3
    cqt_raw = np.abs(librosa.cqt(
        y=y_harmonic, sr=sr, fmin=fmin, hop_length=hop_length,
        n_bins=n_semitones * bins_per_semi,
        bins_per_octave=12 * bins_per_semi,
    ))

    # Collapse to 1 bin per semitone (take max of each group)
    cqt_semitone = np.zeros((n_semitones, cqt_raw.shape[1]))
    for i in range(n_semitones):
        cqt_semitone[i] = np.max(cqt_raw[i * bins_per_semi:(i + 1) * bins_per_semi], axis=0)

    # Build note names for each semitone bin (use standard A=440 for naming)
    bin_notes = []
    bin_freqs = []
    fmin_standard = librosa.note_to_hz("E2")
    for i in range(n_semitones):
        midi_note = librosa.hz_to_midi(fmin_standard) + i
        bin_notes.append(librosa.midi_to_note(midi_note))
        bin_freqs.append(float(librosa.midi_to_hz(midi_note)))

    # ------------------------------------------------------------------
    # Helper: check if CQT bin idx is a local spectral peak (higher than
    # both immediate neighbours).  Spectral leakage produces monotonically
    # rising/falling energy around the true note — only the true note is a
    # peak.
    # ------------------------------------------------------------------
    def _is_peak(mag, idx):
        left = mag[idx - 1] if idx > 0 else 0.0
        right = mag[idx + 1] if idx < n_semitones - 1 else 0.0
        return mag[idx] > left and mag[idx] > right

    # Harmonic intervals (semitones): octave through 8th harmonic
    # 12=oct, 19=oct+5th, 24=2oct, 28=2oct+3rd, 31=2oct+5th, 34=2oct+m7, 36=3oct
    harm_intervals = {12, 19, 24, 28, 31, 34, 36}

    def _harmonic_set(root_idx):
        """Return set of bins that are exact harmonics of root_idx.
        No tolerance needed — tuning is already corrected in CQT fmin."""
        return {root_idx + h for h in harm_intervals}

    # ------------------------------------------------------------------
    # Onset detection
    # ------------------------------------------------------------------
    onset_frames = librosa.onset.onset_detect(
        y=y_harmonic, sr=sr, hop_length=hop_length, backtrack=True,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    # Analysis window — short (≈150 ms) to capture the attack transient where
    # the fundamental is strongest relative to harmonics
    window_sec = 0.25
    window_frames = max(1, int(window_sec * sr / hop_length))

    # Global threshold: 50th percentile of non-zero CQT energy
    all_mags = cqt_semitone[cqt_semitone > 0]
    if len(all_mags) == 0:
        return []
    global_threshold = np.percentile(all_mags, 50)

    # ------------------------------------------------------------------
    # Per-onset note extraction using local-peak + harmonic suppression
    # ------------------------------------------------------------------
    raw_events = []  # list of (time, {note_name: (freq, mag)})

    for onset_t in onset_times:
        onset_frame = librosa.time_to_frames(onset_t, sr=sr, hop_length=hop_length)
        start = onset_frame
        end = min(start + window_frames, cqt_semitone.shape[1])
        if start >= cqt_semitone.shape[1]:
            continue

        # Average energy per semitone over the attack window
        mag_window = np.mean(cqt_semitone[:, start:end], axis=1)
        max_mag = np.max(mag_window)
        if max_mag < global_threshold:
            continue

        # Two-pass peak detection:
        # 1) Strong threshold for bass candidates (20% of max)
        # 2) Lower threshold for treble melody (10% of max)
        # The local peak filter handles noise — we can afford a lower threshold
        treble_start = 24  # E4 and above
        strong_thresh = max(max_mag * 0.20, global_threshold)
        melody_thresh = max(max_mag * 0.10, global_threshold)

        active_strong = np.where(mag_window > strong_thresh)[0]
        active_melody = np.where(
            (mag_window > melody_thresh) &
            (np.arange(n_semitones) >= treble_start)
        )[0]
        active = np.unique(np.concatenate([active_strong, active_melody]))
        if len(active) == 0:
            continue

        # ---- LOCAL PEAK FILTER ----
        # Only keep bins that are spectral peaks (eliminates leakage)
        peaks = [idx for idx in active if _is_peak(mag_window, idx)]
        if not peaks:
            continue

        # ---- BASS = lowest peak ----
        bass_idx = peaks[0]
        bass_harmonics = _harmonic_set(bass_idx)

        # ---- Keep peaks NOT explained as harmonics of the bass ----
        # Exception: if the "harmonic" is stronger than the bass, it's a real note
        bass_mag = mag_window[bass_idx]
        real_bins = [bass_idx]
        for idx in peaks[1:]:
            if idx in bass_harmonics and mag_window[idx] < bass_mag * 2.0:
                continue  # weaker than 2x bass — likely a harmonic
            real_bins.append(idx)

        # ---- For each real note, suppress ITS harmonics ----
        # Only suppress if the harmonic is weaker than 2x the fundamental
        final_bins = []
        suppressed = {}  # bin -> fundamental magnitude
        for idx in sorted(real_bins):
            if idx in suppressed and mag_window[idx] < suppressed[idx] * 2.0:
                continue
            final_bins.append(idx)
            for h in harm_intervals:
                target = idx + h
                if target < n_semitones:
                    suppressed[target] = mag_window[idx]

        # Build frame note dict
        frame_notes = {}
        for idx in final_bins:
            note = bin_notes[idx]
            freq = bin_freqs[idx]
            mag = float(mag_window[idx])
            if note not in frame_notes or mag > frame_notes[note][1]:
                frame_notes[note] = (freq, mag)

        if frame_notes:
            raw_events.append((onset_t, frame_notes))

    if not raw_events:
        return []

    # ------------------------------------------------------------------
    # Sustained-note detection: measure how long each note rings so we
    # can set proper MIDI durations. We DON'T create new note events —
    # just record the sustain end time per onset note.
    # ------------------------------------------------------------------
    sustain_check_interval = 0.2  # seconds between checks
    sustain_ratio = 0.20
    # sustain_ends: {(event_index, note_name): end_time}
    sustain_ends = {}

    for ev_i in range(len(raw_events)):
        t_start = raw_events[ev_i][0]
        prev_notes = raw_events[ev_i][1]

        for note_name, (freq, onset_mag) in prev_notes.items():
            # Find which bin this note corresponds to
            if note_name not in bin_notes:
                continue
            note_bin = bin_notes.index(note_name)

            # Scan forward to find when the note dies
            check_t = t_start + sustain_check_interval
            last_alive = t_start
            while check_t < t_start + 4.0:  # max 4 seconds sustain
                check_frame = librosa.time_to_frames(check_t, sr=sr, hop_length=hop_length)
                c_end = min(check_frame + window_frames, cqt_semitone.shape[1])
                if check_frame >= cqt_semitone.shape[1]:
                    break
                mag_at_check = np.mean(cqt_semitone[note_bin, check_frame:c_end])
                if mag_at_check >= onset_mag * sustain_ratio:
                    last_alive = check_t
                else:
                    break
                check_t += sustain_check_interval

            sustain_ends[(ev_i, note_name)] = last_alive + sustain_check_interval

    # ------------------------------------------------------------------
    # Build a lookup for sustain end time per (event_index, note_name)
    # Map to (onset_time, note_name) -> sustain_end for easy access later
    # ------------------------------------------------------------------
    sustain_by_note = {}  # (onset_time, note_name) -> end_time
    for (ev_i, note_name), end_t in sustain_ends.items():
        onset_t = raw_events[ev_i][0]
        sustain_by_note[(round(onset_t, 3), note_name)] = end_t

    # ------------------------------------------------------------------
    # Group nearby onsets into single chord events (arpeggios / strums)
    # ------------------------------------------------------------------
    group_window = 0.08  # 80ms

    grouped = []
    current_time = raw_events[0][0]
    current_notes = dict(raw_events[0][1])

    for t, note_dict in raw_events[1:]:
        if t - current_time < group_window:
            for note, (freq, mag) in note_dict.items():
                if note not in current_notes or mag > current_notes[note][1]:
                    current_notes[note] = (freq, mag)
        else:
            grouped.append((current_time, current_notes))
            current_time = t
            current_notes = dict(note_dict)
    grouped.append((current_time, current_notes))

    # ------------------------------------------------------------------
    # Normalize magnitudes to MIDI velocity range (50–127)
    # Deduplicate: if the same note appears at consecutive onsets, keep
    # only the first occurrence and let it ring via sustain duration.
    # ------------------------------------------------------------------
    all_event_mags = [mag for _, nd in grouped for _, (_, mag) in nd.items()]
    mag_min = min(all_event_mags)
    mag_max = max(all_event_mags)
    mag_range = mag_max - mag_min if mag_max > mag_min else 1.0

    # Track which notes are currently "ringing" so we don't re-trigger
    ringing = {}  # note_name -> (end_time, index in notes list, onset_time)

    notes = []
    for t, notes_dict in grouped:
        # Snapshot of notes ringing from previous onsets (not this one)
        ringing_before = {k: v for k, v in ringing.items() if t < v[0]}

        for note, (freq, mag) in sorted(notes_dict.items(), key=lambda x: x[1][0]):
            # Check if this note is still ringing from a previous onset
            if note in ringing and t < ringing[note][0]:
                prev_onset_t = ringing[note][2]
                # Too close to previous onset — same attack, not a re-pluck
                if t - prev_onset_t < 0.10:
                    continue
                # Note is still ringing — but check for a genuine re-pluck.
                # Compare CQT energy at onset vs just before: a spike means
                # the string was plucked again.
                note_bin = bin_notes.index(note) if note in bin_notes else None
                if note_bin is not None:
                    onset_frame = librosa.time_to_frames(t, sr=sr, hop_length=hop_length)
                    pre_start = max(0, onset_frame - 3)
                    if onset_frame > pre_start:
                        pre_mag = float(np.mean(cqt_semitone[note_bin, pre_start:onset_frame]))
                    else:
                        pre_mag = 0.0
                    if pre_mag > 0 and mag <= pre_mag * 1.5:
                        continue  # energy is flat — still sustaining, not re-plucked
                    # Re-pluck detected: truncate previous note's duration
                    prev_idx = ringing[note][1]
                    old = notes[prev_idx]
                    notes[prev_idx] = (old[0], old[1], old[2], old[3], max(t - old[0], 0.12))
                else:
                    continue

            # Suppress high-register harmonics of notes ringing from previous
            # onsets.  Only applies above bin 36 (E5+) — real guitar notes
            # rarely reach that range, but harmonic artifacts are common there.
            note_bin = bin_notes.index(note) if note in bin_notes else None
            if note_bin is not None and note_bin >= 36:
                is_harmonic_of_ringing = False
                for r_note, (r_end, _, _) in ringing_before.items():
                    r_bin = bin_notes.index(r_note) if r_note in bin_notes else None
                    if r_bin is not None and (note_bin - r_bin) in harm_intervals:
                        is_harmonic_of_ringing = True
                        break
                if is_harmonic_of_ringing:
                    continue

            normalized = (mag - mag_min) / mag_range
            velocity = int(50 + 77 * (normalized ** 0.35))

            # Get sustain duration from our measurement
            sustain_end = sustain_by_note.get((round(t, 3), note))
            if sustain_end and sustain_end > t:
                dur = sustain_end - t
            else:
                dur = 0.3  # default short duration
            dur = max(dur, 0.12)  # minimum 120ms

            note_idx = len(notes)
            notes.append((t, note, freq, velocity, dur))
            ringing[note] = (t + dur, note_idx, t)

    return notes


def export_midi(filepath, bpm, notes, chords):
    """Export detected notes and chords to a MIDI file."""
    midi = MIDIFile(2, deinterleave=False)  # 2 tracks: melody + chords

    beats_per_sec = bpm / 60.0

    # Track 0: Melody
    midi.addTrackName(0, 0, "Melody")
    midi.addTempo(0, 0, bpm)
    midi.addProgramChange(0, 0, 0, 25)  # GM: Acoustic Guitar (nylon)

    for time_stamp, note, freq, velocity, dur_sec in notes:
        midi_note = librosa.note_to_midi(note)
        start_beat = time_stamp * beats_per_sec
        dur_beats = max(dur_sec * beats_per_sec, 0.25)

        midi.addNote(0, 0, midi_note, start_beat, dur_beats, velocity)

    # Track 1: Chords
    midi.addTrackName(1, 0, "Chords")
    midi.addTempo(1, 0, bpm)
    midi.addProgramChange(1, 1, 0, 25)

    CHORD_MIDI = {}
    for i, note_name in enumerate(CHROMA_NOTES):
        base = 48 + i  # C3 octave
        CHORD_MIDI[note_name] = [base, base + 4, base + 7]          # major
        CHORD_MIDI[f"{note_name}m"] = [base, base + 3, base + 7]    # minor

    for i, (time_stamp, chord) in enumerate(chords):
        if chord not in CHORD_MIDI:
            continue
        start_beat = time_stamp * beats_per_sec

        if i + 1 < len(chords):
            dur_sec = chords[i + 1][0] - time_stamp
        else:
            dur_sec = 2.0
        dur_beats = max(dur_sec * beats_per_sec, 0.5)

        for midi_note in CHORD_MIDI[chord]:
            midi.addNote(1, 1, midi_note, start_beat, dur_beats, 80)

    # Write file
    midi_path = os.path.splitext(filepath)[0] + ".mid"
    with open(midi_path, "wb") as f:
        midi.writeFile(f)

    return midi_path


def format_time(seconds):
    """Format seconds as M:SS."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def analyze(filepath):
    """Run full analysis on an audio file."""
    print(f"\nAnalyzing: {os.path.basename(filepath)}")
    print("Loading audio...", end=" ", flush=True)

    y, sr = librosa.load(filepath, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"({format_time(duration)})\n")

    # Tempo
    print("Detecting tempo...", end=" ", flush=True)
    bpm = detect_tempo(y, sr)
    print(f"{bpm} BPM\n")

    # Key
    print("Detecting key...", end=" ", flush=True)
    key = detect_key(y, sr)
    print(f"{key}\n")

    # Chords
    print("Detecting chords...", end=" ", flush=True)
    chords = detect_chords(y, sr)
    print(f"{len(chords)} changes\n")

    # Notes
    print("Detecting notes...", end=" ", flush=True)
    notes = detect_notes(y, sr, bpm)
    print(f"{len(notes)} notes\n")

    # --- Summary ---
    print("=" * 50)
    print(f"  Song:   {os.path.splitext(os.path.basename(filepath))[0]}")
    print(f"  Tempo:  {bpm} BPM")
    print(f"  Key:    {key}")
    print(f"  Length: {format_time(duration)}")
    print("=" * 50)

    print("\n  Chord Progression:\n")
    for time_stamp, chord in chords:
        print(f"    {format_time(time_stamp):>5s}  {chord}")

    print("\n  Melody Notes:\n")
    for time_stamp, note, freq, velocity, dur in notes:
        bar = "#" * (velocity // 10)
        print(f"    {format_time(time_stamp):>5s}  {note:<5s} vel:{velocity:>3d}  dur:{dur:.2f}s  {bar}")

    # MIDI export
    print("\nExporting MIDI...", end=" ", flush=True)
    midi_path = export_midi(filepath, bpm, notes, chords)
    print(f"saved to {os.path.basename(midi_path)}\n")

    return {
        "bpm": bpm,
        "key": key,
        "duration": duration,
        "chords": chords,
        "notes": notes,
        "midi": midi_path,
    }


if __name__ == "__main__":
    # Allow running standalone with a file path
    if len(sys.argv) > 1:
        analyze(sys.argv[1])
    else:
        # Import from extract to use the song picker
        from extract import pick_song
        filepath = pick_song()
        analyze(filepath)
