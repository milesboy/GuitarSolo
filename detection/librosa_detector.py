"""Librosa CQT-based polyphonic note detection.

Extracted from analyze.py. Supports tunable parameters for the
optimization loop.
"""
import librosa
import numpy as np

CHROMA_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def detect_tempo(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(np.round(tempo[0] if hasattr(tempo, '__len__') else tempo))


def detect_key(y, sr):
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
    CHORD_TEMPLATES = {}
    for i, note in enumerate(CHROMA_NOTES):
        major = np.zeros(12)
        major[i] = 1
        major[(i + 4) % 12] = 1
        major[(i + 7) % 12] = 1
        CHORD_TEMPLATES[note] = major
        minor = np.zeros(12)
        minor[i] = 1
        minor[(i + 3) % 12] = 1
        minor[(i + 7) % 12] = 1
        CHORD_TEMPLATES[f"{note}m"] = minor

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(range(chroma.shape[1]), sr=sr, hop_length=hop_length)
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

    collapsed = [chords[0]]
    for time_stamp, chord in chords[1:]:
        if chord != collapsed[-1][1]:
            collapsed.append((time_stamp, chord))
    return collapsed


def detect_notes(y, sr, bpm=120.0,
                 hop_length=512,
                 strong_thresh_pct=0.20,
                 melody_thresh_pct=0.10,
                 sustain_ratio=0.20,
                 group_window=0.08):
    """Detect notes with tunable parameters for optimization loop.

    Returns list of (time, note_name, freq, velocity, duration).
    """
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
        cqt_semitone[i] = np.max(cqt_raw[i * bins_per_semi:(i + 1) * bins_per_semi], axis=0)

    bin_notes = []
    bin_freqs = []
    fmin_standard = librosa.note_to_hz("E2")
    for i in range(n_semitones):
        midi_note = librosa.hz_to_midi(fmin_standard) + i
        bin_notes.append(librosa.midi_to_note(midi_note))
        bin_freqs.append(float(librosa.midi_to_hz(midi_note)))

    def _is_peak(mag, idx):
        left = mag[idx - 1] if idx > 0 else 0.0
        right = mag[idx + 1] if idx < n_semitones - 1 else 0.0
        return mag[idx] > left and mag[idx] > right

    harm_intervals = {12, 19, 24, 28, 31, 34, 36}

    def _harmonic_set(root_idx):
        return {root_idx + h for h in harm_intervals}

    onset_frames = librosa.onset.onset_detect(
        y=y_harmonic, sr=sr, hop_length=hop_length, backtrack=True,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    window_sec = 0.25
    window_frames = max(1, int(window_sec * sr / hop_length))

    all_mags = cqt_semitone[cqt_semitone > 0]
    if len(all_mags) == 0:
        return []
    global_threshold = np.percentile(all_mags, 50)

    raw_events = []
    treble_start = 24

    for onset_t in onset_times:
        onset_frame = librosa.time_to_frames(onset_t, sr=sr, hop_length=hop_length)
        start = onset_frame
        end = min(start + window_frames, cqt_semitone.shape[1])
        if start >= cqt_semitone.shape[1]:
            continue

        mag_window = np.mean(cqt_semitone[:, start:end], axis=1)
        max_mag = np.max(mag_window)
        if max_mag < global_threshold:
            continue

        strong_thresh = max(max_mag * strong_thresh_pct, global_threshold)
        melody_thresh = max(max_mag * melody_thresh_pct, global_threshold)

        active_strong = np.where(mag_window > strong_thresh)[0]
        active_melody = np.where(
            (mag_window > melody_thresh) &
            (np.arange(n_semitones) >= treble_start)
        )[0]
        active = np.unique(np.concatenate([active_strong, active_melody]))
        if len(active) == 0:
            continue

        peaks = [idx for idx in active if _is_peak(mag_window, idx)]
        if not peaks:
            continue

        bass_idx = peaks[0]
        bass_harmonics = _harmonic_set(bass_idx)
        bass_mag = mag_window[bass_idx]
        real_bins = [bass_idx]
        for idx in peaks[1:]:
            if idx in bass_harmonics and mag_window[idx] < bass_mag * 2.0:
                continue
            real_bins.append(idx)

        final_bins = []
        suppressed = {}
        for idx in sorted(real_bins):
            if idx in suppressed and mag_window[idx] < suppressed[idx] * 2.0:
                continue
            final_bins.append(idx)
            for h in harm_intervals:
                target = idx + h
                if target < n_semitones:
                    suppressed[target] = mag_window[idx]

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

    # Sustained-note detection
    sustain_check_interval = 0.2
    sustain_ends = {}

    for ev_i in range(len(raw_events)):
        t_start = raw_events[ev_i][0]
        prev_notes = raw_events[ev_i][1]
        for note_name, (freq, onset_mag) in prev_notes.items():
            if note_name not in bin_notes:
                continue
            note_bin = bin_notes.index(note_name)
            check_t = t_start + sustain_check_interval
            last_alive = t_start
            while check_t < t_start + 4.0:
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

    sustain_by_note = {}
    for (ev_i, note_name), end_t in sustain_ends.items():
        onset_t = raw_events[ev_i][0]
        sustain_by_note[(round(onset_t, 3), note_name)] = end_t

    # Group nearby onsets
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

    # Normalize magnitudes to velocity
    all_event_mags = [mag for _, nd in grouped for _, (_, mag) in nd.items()]
    mag_min = min(all_event_mags)
    mag_max = max(all_event_mags)
    mag_range = mag_max - mag_min if mag_max > mag_min else 1.0

    ringing = {}  # note_name -> (end_time, index, onset_time)

    notes = []
    for t, notes_dict in grouped:
        ringing_before = {k: v for k, v in ringing.items() if t < v[0]}

        for note, (freq, mag) in sorted(notes_dict.items(), key=lambda x: x[1][0]):
            if note in ringing and t < ringing[note][0]:
                prev_onset_t = ringing[note][2]
                if t - prev_onset_t < 0.10:
                    continue
                note_bin = bin_notes.index(note) if note in bin_notes else None
                if note_bin is not None:
                    onset_frame = librosa.time_to_frames(t, sr=sr, hop_length=hop_length)
                    pre_start = max(0, onset_frame - 3)
                    if onset_frame > pre_start:
                        pre_mag = float(np.mean(cqt_semitone[note_bin, pre_start:onset_frame]))
                    else:
                        pre_mag = 0.0
                    if pre_mag > 0 and mag <= pre_mag * 1.5:
                        continue
                    prev_idx = ringing[note][1]
                    old = notes[prev_idx]
                    notes[prev_idx] = (old[0], old[1], old[2], old[3], max(t - old[0], 0.12))
                else:
                    continue

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

            sustain_end = sustain_by_note.get((round(t, 3), note))
            if sustain_end and sustain_end > t:
                dur = sustain_end - t
            else:
                dur = 0.3
            dur = max(dur, 0.12)

            note_idx = len(notes)
            notes.append((t, note, freq, velocity, dur))
            ringing[note] = (t + dur, note_idx, t)

    return notes
