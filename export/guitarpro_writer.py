"""Write Guitar Pro (.gp5) files.

Track 1: Melody/fingerpicking with detected notes
Timing: 16th-note quantization grid. Each measure sums to exactly
16 sixteenth notes.
"""
import guitarpro
import librosa
import math

from articulation.detector import ArticulationType

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
    mt.number = 1
    mt.port = 0
    mt.offset = 0
    mt.channel.channel = 0
    mt.channel.instrument = 25
    mt.isPercussionTrack = False
    mt.strings = [
        guitarpro.GuitarString(1, 64),
        guitarpro.GuitarString(2, 59),
        guitarpro.GuitarString(3, 55),
        guitarpro.GuitarString(4, 50),
        guitarpro.GuitarString(5, 45),
        guitarpro.GuitarString(6, 40),
    ]

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

    # Group onsets by measure
    onsets_by_measure = {}
    for onset in sorted(note_groups.keys()):
        m_idx = int(onset / spm)
        if m_idx not in onsets_by_measure:
            onsets_by_measure[m_idx] = []
        onsets_by_measure[m_idx].append(onset)

    # Fill each measure
    for m_idx in range(num_measures):
        if m_idx >= len(mt.measures):
            break
        voice = mt.measures[m_idx].voices[0]
        m_start = m_idx * spm
        onsets = onsets_by_measure.get(m_idx, [])
        cursor = 0

        if not onsets:
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

            # Duration until next note or measure end
            if j + 1 < len(onsets):
                next_slot = round((onsets[j + 1] - m_start) / sec_per_16th)
                next_slot = max(0, min(next_slot, SLOTS_PER_MEASURE))
                avail = max(1, next_slot - cursor)
            else:
                avail = SLOTS_PER_MEASURE - cursor
            avail = min(avail, SLOTS_PER_MEASURE - cursor)
            avail = max(avail, 1)

            note_dur = _snap_down(avail)

            beat = guitarpro.Beat(
                voice, status=guitarpro.BeatStatus.normal)
            beat.duration = guitarpro.Duration(value=DUR_MAP[note_dur])

            for (nd, art) in note_groups[onset]:
                note = guitarpro.Note(beat)
                note.string = 6 - nd[5]
                note.value = nd[6]
                note.type = guitarpro.NoteType.normal
                note.velocity = min(max(nd[3], 1), 127)
                beat.notes.append(note)

            voice.beats.append(beat)
            cursor += note_dur

            # Fill leftover from snap
            leftover = avail - note_dur
            if leftover > 0:
                _add_rests(voice, leftover)
                cursor += leftover

        # Fill remainder of measure
        if cursor < SLOTS_PER_MEASURE:
            _add_rests(voice, SLOTS_PER_MEASURE - cursor)

    guitarpro.write(song, output_path)
    return output_path
