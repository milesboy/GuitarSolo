"""Write Guitar Pro (.gp5) files.

Track 1: Melody/fingerpicking with detected notes
Timing: 16th-note quantization grid. Each measure sums to exactly
16 sixteenth notes. Notes with sustain past the measure boundary
are tied into the next measure.
"""
import guitarpro
import librosa
import math

from articulation.detector import ArticulationType

# Voice ranges for independent sustain
BASS_MAX = 55  # E2-G3

# 16th-note grid
SLOTS_PER_MEASURE = 16
DUR_MAP = {16: 1, 8: 2, 4: 4, 2: 8, 1: 16}


def _snap_down(slots):
    for s in [16, 8, 4, 2, 1]:
        if s <= slots:
            return s
    return 1


def _add_rests(voice, slots):
    rem = slots
    while rem > 0:
        s = _snap_down(rem)
        b = guitarpro.Beat(voice, status=guitarpro.BeatStatus.rest)
        b.duration = guitarpro.Duration(value=DUR_MAP[s])
        voice.beats.append(b)
        rem -= s


def _add_ties(voice, slots, source_beat):
    """Add tied beats continuing from source_beat."""
    rem = slots
    while rem > 0:
        s = _snap_down(rem)
        beat = guitarpro.Beat(voice, status=guitarpro.BeatStatus.normal)
        beat.duration = guitarpro.Duration(value=DUR_MAP[s])
        for src_n in source_beat.notes:
            n = guitarpro.Note(beat)
            n.string = src_n.string
            n.value = src_n.value
            n.type = guitarpro.NoteType.tie
            n.velocity = src_n.velocity
            beat.notes.append(n)
        voice.beats.append(beat)
        rem -= s


def _note_range(note_data):
    """Classify a note as bass or melody by MIDI pitch."""
    try:
        midi = librosa.note_to_midi(
            note_data[1].replace("\u266f", "#").replace("\u266d", "b"))
        return "bass" if midi <= BASS_MAX else "melody"
    except Exception:
        return "melody"


def _group_ranges(group):
    """Get the set of ranges present in a note group."""
    return {_note_range(nd) for nd, _ in group}


def write_guitarpro(fretted_notes, articulations, bpm, key="C major",
                    chords=None, title="", artist="",
                    output_path="output.gp5"):
    """Create a Guitar Pro file from detected notes."""
    song = guitarpro.Song()
    song.title = title or "Untitled"
    song.artist = artist or ""
    song.tempo = int(bpm)

    bps = bpm / 60.0
    spm = 4.0 / bps
    sec_per_16th = 60.0 / bpm / 4

    mt = song.tracks[0]
    mt.name = "Guitar"
    mt.channel.instrument = 25

    if not fretted_notes:
        guitarpro.write(song, output_path)
        return output_path

    max_time = max(n[0] + n[4] for n in fretted_notes)
    num_measures = max(1, math.ceil(max_time / spm) + 2)

    while len(song.measureHeaders) < num_measures:
        h = guitarpro.MeasureHeader()
        song.measureHeaders.append(h)
        mt.measures.append(guitarpro.Measure(mt, h))

    # Group notes by onset time
    note_groups = {}
    for i, n in enumerate(fretted_notes):
        t = round(n[0], 3)
        if t not in note_groups:
            note_groups[t] = []
        art = (articulations[i]
               if i < len(articulations)
               else ArticulationType.PICKED)
        note_groups[t].append((n, art))

    sorted_onsets = sorted(note_groups.keys())

    # Pre-compute ranges for range-aware sustain
    onset_ranges = {t: _group_ranges(note_groups[t]) for t in sorted_onsets}

    # Group onsets by measure
    onsets_by_measure = {}
    for onset in sorted_onsets:
        m_idx = int(onset / spm)
        if m_idx not in onsets_by_measure:
            onsets_by_measure[m_idx] = []
        onsets_by_measure[m_idx].append(onset)

    # Fill each measure with notes, rests, and ties
    carry = None  # (overflow_slots, source_beat)

    for m_idx in range(num_measures):
        if m_idx >= len(mt.measures):
            break
        voice = mt.measures[m_idx].voices[0]
        m_start = m_idx * spm
        onsets = onsets_by_measure.get(m_idx, [])
        cursor = 0

        # Carry-in: tied notes from previous measure
        if carry is not None:
            overflow, src_beat = carry
            carry = None
            first_slot = SLOTS_PER_MEASURE
            if onsets:
                first_slot = round((onsets[0] - m_start) / sec_per_16th)
                first_slot = max(0, min(first_slot, SLOTS_PER_MEASURE))
            tie_slots = min(overflow, first_slot)
            if tie_slots > 0:
                _add_ties(voice, tie_slots, src_beat)
                cursor = tie_slots

        if not onsets and cursor == 0:
            b = guitarpro.Beat(voice, status=guitarpro.BeatStatus.rest)
            b.duration = guitarpro.Duration(value=1)
            voice.beats.append(b)
            continue

        for j, onset in enumerate(onsets):
            if cursor >= SLOTS_PER_MEASURE:
                break
            slot = round((onset - m_start) / sec_per_16th)
            slot = max(0, min(slot, SLOTS_PER_MEASURE - 1))
            slot = max(slot, cursor)
            if slot >= SLOTS_PER_MEASURE:
                break

            # Rest before note
            if slot > cursor:
                _add_rests(voice, slot - cursor)
                cursor = slot

            # Compute ring duration (range-aware)
            group = note_groups[onset]
            sustain = max(nd[4] for nd, _ in group)
            my_ranges = onset_ranges[onset]
            gap = None
            idx = sorted_onsets.index(onset)
            for fi in range(idx + 1, len(sorted_onsets)):
                if my_ranges & onset_ranges[sorted_onsets[fi]]:
                    gap = sorted_onsets[fi] - onset
                    break
            ring_sec = min(sustain, gap) if gap else sustain
            ring_slots = max(1, round(ring_sec / sec_per_16th))

            remaining = SLOTS_PER_MEASURE - cursor

            # Cap at next note in this measure
            avail = ring_slots
            if j + 1 < len(onsets):
                next_slot = round((onsets[j + 1] - m_start) / sec_per_16th)
                next_slot = max(0, min(next_slot, SLOTS_PER_MEASURE))
                avail = min(avail, next_slot - cursor)
            avail = max(avail, 1)

            note_dur = _snap_down(min(avail, remaining))

            # Create the note beat
            beat = guitarpro.Beat(
                voice, status=guitarpro.BeatStatus.normal)
            beat.duration = guitarpro.Duration(value=DUR_MAP[note_dur])

            for (nd, art) in group:
                note = guitarpro.Note(beat)
                note.string = 6 - nd[5]
                note.value = nd[6]
                note.type = guitarpro.NoteType.normal
                note.velocity = min(max(nd[3], 1), 127)
                beat.notes.append(note)

            voice.beats.append(beat)
            cursor += note_dur

            # Fill leftover from snap
            leftover = min(avail, remaining) - note_dur
            if leftover > 0:
                _add_rests(voice, leftover)
                cursor += leftover

            # Carry tie if last note in measure and sustain overflows
            if j + 1 >= len(onsets) and ring_slots > remaining:
                carry = (ring_slots - remaining, beat)

        # Fill remainder of measure
        if cursor < SLOTS_PER_MEASURE:
            _add_rests(voice, SLOTS_PER_MEASURE - cursor)

    guitarpro.write(song, output_path)
    return output_path
