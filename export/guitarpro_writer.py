"""Write Guitar Pro (.gp5) files with articulation support.

Track 1: Melody/fingerpicking (detected notes with articulations)
Track 2: Chords (muted by default — reference track)

Timing: 16th-note quantization grid. Each measure sums to exactly 16
sixteenth notes. Notes ring for their detected sustain duration with
range-aware capping (bass and melody sustain independently). Ties
carry notes across bar lines.
"""
import guitarpro
from guitarpro.models import BendPoint
import librosa
import math

from articulation.detector import ArticulationType

# Voice ranges for independent sustain
BASS_RANGE = (40, 55)    # E2 to G3
MELODY_RANGE = (56, 88)  # G#3 to E6

# 16th-note grid: 16 slots per measure in 4/4
SLOTS_PER_MEASURE = 16

# Slot count -> GP Duration.value
DUR_MAP = {16: 1, 8: 2, 4: 4, 2: 8, 1: 16}

# Chord voicings as (gp_string, fret)
CHORD_VOICINGS = {
    "C":  [(1,0),(2,1),(3,0),(4,2),(5,3)],
    "D":  [(1,2),(2,3),(3,2),(4,0)],
    "Dm": [(1,1),(2,3),(3,2),(4,0)],
    "E":  [(1,0),(2,0),(3,1),(4,2),(5,2),(6,0)],
    "Em": [(1,0),(2,0),(3,0),(4,2),(5,2),(6,0)],
    "F":  [(1,1),(2,1),(3,2),(4,3),(5,3),(6,1)],
    "Fm": [(1,1),(2,1),(3,1),(4,3),(5,3),(6,1)],
    "G":  [(1,3),(2,0),(3,0),(4,0),(5,2),(6,3)],
    "A":  [(1,0),(2,2),(3,2),(4,2),(5,0)],
    "Am": [(1,0),(2,1),(3,2),(4,2),(5,0)],
    "B":  [(1,2),(2,4),(3,4),(4,4),(5,2)],
    "Bm": [(1,2),(2,3),(3,4),(4,4),(5,2)],
    "C#": [(1,1),(2,2),(3,3),(4,3),(5,4)],
    "C#m":[(1,0),(2,2),(3,2),(4,1),(5,4)],
    "D#": [(1,3),(2,4),(3,3),(4,1),(5,6)],
    "D#m":[(1,2),(2,4),(3,3),(4,1),(5,6)],
    "F#": [(1,2),(2,2),(3,3),(4,4),(5,4),(6,2)],
    "F#m":[(1,2),(2,2),(3,2),(4,4),(5,4),(6,2)],
    "G#": [(1,4),(2,4),(3,5),(4,6),(5,6),(6,4)],
    "G#m":[(1,4),(2,4),(3,4),(4,6),(5,6),(6,4)],
    "A#": [(1,1),(2,1),(3,3),(4,3),(5,1)],
    "A#m":[(1,1),(2,1),(3,2),(4,3),(5,1)],
}
CHORD_VOICINGS["Db"] = CHORD_VOICINGS["C#"]
CHORD_VOICINGS["Dbm"] = CHORD_VOICINGS["C#m"]
CHORD_VOICINGS["Eb"] = CHORD_VOICINGS["D#"]
CHORD_VOICINGS["Ebm"] = CHORD_VOICINGS["D#m"]
CHORD_VOICINGS["Gb"] = CHORD_VOICINGS["F#"]
CHORD_VOICINGS["Gbm"] = CHORD_VOICINGS["F#m"]
CHORD_VOICINGS["Ab"] = CHORD_VOICINGS["G#"]
CHORD_VOICINGS["Abm"] = CHORD_VOICINGS["G#m"]
CHORD_VOICINGS["Bb"] = CHORD_VOICINGS["A#"]
CHORD_VOICINGS["Bbm"] = CHORD_VOICINGS["A#m"]


def _note_range(note_name):
    """Classify a note into bass or melody range."""
    try:
        midi = librosa.note_to_midi(
            note_name.replace("\u266f", "#").replace("\u266d", "b"))
    except Exception:
        return "melody"
    if midi <= BASS_RANGE[1]:
        return "bass"
    return "melody"


def _group_range(group):
    """Determine the range(s) present in a note group."""
    ranges = set()
    for (note_data, _art) in group:
        ranges.add(_note_range(note_data[1]))
    return ranges


# --- Beat builders ---

def _snap_down(slots):
    """Largest power-of-2 duration that fits in slots."""
    for s in [16, 8, 4, 2, 1]:
        if s <= slots:
            return s
    return 1


def _add_rests(voice, slots):
    """Decompose slots into exact rest beats (e.g. 5 = 4+1)."""
    rem = slots
    while rem > 0:
        for s in [16, 8, 4, 2, 1]:
            if s <= rem:
                b = guitarpro.Beat(voice, status=guitarpro.BeatStatus.rest)
                b.duration = guitarpro.Duration(value=DUR_MAP[s])
                voice.beats.append(b)
                rem -= s
                break


def _add_note(voice, slots, group):
    """Add a note beat snapped to a clean power-of-2 duration.

    Returns (actual_slots_used, beat_object).
    """
    dur = _snap_down(slots)
    beat = guitarpro.Beat(voice, status=guitarpro.BeatStatus.normal)
    beat.duration = guitarpro.Duration(value=DUR_MAP[dur])
    for (nd, art) in group:
        note = guitarpro.Note(beat)
        note.string = 6 - nd[5]   # string index -> GP string number
        note.value = nd[6]         # fret
        note.type = guitarpro.NoteType.normal
        note.velocity = min(max(nd[3], 1), 127)
        note.effect = guitarpro.NoteEffect()

        if art == ArticulationType.HAMMER_ON:
            note.effect.hammer = True
        elif art == ArticulationType.PULL_OFF:
            note.effect.hammer = True
        elif art in (ArticulationType.SLIDE_UP, ArticulationType.SLIDE_DOWN):
            note.effect.slides = [guitarpro.SlideType.shiftSlideTo]
        elif art == ArticulationType.BEND:
            bend_effect = guitarpro.BendEffect()
            bend_effect.type = guitarpro.BendType.bend
            bend_effect.value = 100
            bend_effect.points = [
                BendPoint(0, 0),
                BendPoint(6, 100),
                BendPoint(12, 100),
            ]
            note.effect.bend = bend_effect

        beat.notes.append(note)
    voice.beats.append(beat)
    return dur, beat


def _add_ties(voice, slots, source_beat):
    """Add tied beats that continue ringing from source_beat."""
    rem = slots
    while rem > 0:
        for s in [16, 8, 4, 2, 1]:
            if s <= rem:
                beat = guitarpro.Beat(
                    voice, status=guitarpro.BeatStatus.normal)
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
                break


# --- Main writer ---

def write_guitarpro(fretted_notes, articulations, bpm, key="C major",
                    chords=None, title="", artist="",
                    output_path="output.gp5"):
    """Create a Guitar Pro file with melody track and muted chord track."""
    song = guitarpro.Song()
    song.title = title or "Untitled"
    song.artist = artist or ""
    song.tempo = int(bpm)

    bps = bpm / 60.0
    spm = 4.0 / bps  # seconds per measure
    sec_per_16th = 60.0 / bpm / 4

    # --- Track 1: Melody ---
    melody_track = song.tracks[0]
    melody_track.name = "Melody"
    melody_track.number = 1
    melody_track.port = 0
    melody_track.offset = 0
    melody_track.channel.channel = 0
    melody_track.channel.instrument = 25
    melody_track.isPercussionTrack = False
    melody_track.strings = [
        guitarpro.GuitarString(1, 64),
        guitarpro.GuitarString(2, 59),
        guitarpro.GuitarString(3, 55),
        guitarpro.GuitarString(4, 50),
        guitarpro.GuitarString(5, 45),
        guitarpro.GuitarString(6, 40),
    ]

    # --- Track 2: Chords (muted) ---
    chord_track = guitarpro.Track(song)
    chord_track.name = "Chords"
    chord_track.number = 2
    chord_track.port = 0
    chord_track.offset = 0
    chord_track.isMute = True
    chord_track.channel.channel = 2
    chord_track.channel.instrument = 25
    chord_track.channel.volume = 80
    chord_track.isPercussionTrack = False
    chord_track.strings = [
        guitarpro.GuitarString(1, 64),
        guitarpro.GuitarString(2, 59),
        guitarpro.GuitarString(3, 55),
        guitarpro.GuitarString(4, 50),
        guitarpro.GuitarString(5, 45),
        guitarpro.GuitarString(6, 40),
    ]

    # Determine measures needed
    max_time = 0.0
    if fretted_notes:
        max_time = max(max_time, max(n[0] + n[4] for n in fretted_notes))
    if chords:
        max_time = max(max_time, chords[-1][0] + 2.0)

    if max_time == 0.0:
        song.tracks.append(chord_track)
        guitarpro.write(song, output_path)
        return output_path

    num_measures = max(1, math.ceil(max_time / spm) + 2)

    while len(song.measureHeaders) < num_measures:
        h = guitarpro.MeasureHeader()
        song.measureHeaders.append(h)
        melody_track.measures.append(
            guitarpro.Measure(melody_track, h))

    for h in song.measureHeaders:
        chord_track.measures.append(
            guitarpro.Measure(chord_track, h))
    song.tracks.append(chord_track)

    # ==========================================
    # Populate Track 1: Melody
    # ==========================================
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
    onset_ranges = {t: _group_range(note_groups[t])
                    for t in sorted_onsets}

    # Group onsets by measure
    onsets_by_measure = {}
    for onset in sorted_onsets:
        m_idx = int(onset / spm)
        if m_idx not in onsets_by_measure:
            onsets_by_measure[m_idx] = []
        onsets_by_measure[m_idx].append(onset)

    carry = None  # (overflow_slots, source_beat) for ties

    for m_idx in range(num_measures):
        if m_idx >= len(melody_track.measures):
            break
        voice = melody_track.measures[m_idx].voices[0]
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
            _add_rests(voice, SLOTS_PER_MEASURE)
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

            if avail <= remaining:
                note_dur, beat = _add_note(voice, min(avail, remaining),
                                           group)
                cursor += note_dur
                leftover = min(avail, remaining) - note_dur
                if leftover > 0:
                    _add_rests(voice, leftover)
                    cursor += leftover
                # Carry if sustain extends past measure (last note only)
                if j + 1 >= len(onsets) and ring_slots > remaining:
                    carry = (ring_slots - remaining, beat)
            else:
                note_dur, beat = _add_note(voice, remaining, group)
                cursor += note_dur
                leftover = remaining - note_dur
                if leftover > 0:
                    _add_rests(voice, leftover)
                    cursor += leftover
                carry = (ring_slots - remaining, beat)

        # Fill remainder
        if cursor < SLOTS_PER_MEASURE:
            _add_rests(voice, SLOTS_PER_MEASURE - cursor)

    # ==========================================
    # Populate Track 2: Chords
    # ==========================================
    if chords:
        onsets_by_m_chord = {}
        for i, (ct, cn) in enumerate(chords):
            m_idx = int(ct / spm)
            if m_idx not in onsets_by_m_chord:
                onsets_by_m_chord[m_idx] = []
            dur = (chords[i + 1][0] - ct) if i + 1 < len(chords) else 2.0
            onsets_by_m_chord[m_idx].append((ct, cn, max(dur, 0.25)))

        for m_idx in range(num_measures):
            if m_idx >= len(chord_track.measures):
                break
            voice = chord_track.measures[m_idx].voices[0]
            m_start = m_idx * spm
            chord_onsets = onsets_by_m_chord.get(m_idx, [])

            if not chord_onsets:
                _add_rests(voice, SLOTS_PER_MEASURE)
                continue

            cursor = 0
            for j, (ct, cn, cdur) in enumerate(chord_onsets):
                if cursor >= SLOTS_PER_MEASURE:
                    break
                voicing = CHORD_VOICINGS.get(cn)
                if not voicing:
                    continue

                slot = round((ct - m_start) / sec_per_16th)
                slot = max(0, min(slot, SLOTS_PER_MEASURE - 1))
                slot = max(slot, cursor)
                if slot >= SLOTS_PER_MEASURE:
                    break

                if slot > cursor:
                    _add_rests(voice, slot - cursor)
                    cursor = slot

                # Duration until next chord or measure end
                if j + 1 < len(chord_onsets):
                    ns = round((chord_onsets[j + 1][0] - m_start) / sec_per_16th)
                    ns = max(0, min(ns, SLOTS_PER_MEASURE))
                    avail = ns - cursor
                else:
                    avail = SLOTS_PER_MEASURE - cursor
                avail = max(avail, 1)

                dur_slots = _snap_down(avail)
                beat = guitarpro.Beat(
                    voice, status=guitarpro.BeatStatus.normal)
                beat.duration = guitarpro.Duration(value=DUR_MAP[dur_slots])
                for (gs, gf) in voicing:
                    note = guitarpro.Note(beat)
                    note.string = gs
                    note.value = gf
                    note.type = guitarpro.NoteType.normal
                    note.velocity = 80
                    beat.notes.append(note)
                voice.beats.append(beat)
                cursor += dur_slots

                leftover = avail - dur_slots
                if leftover > 0:
                    _add_rests(voice, leftover)
                    cursor += leftover

            if cursor < SLOTS_PER_MEASURE:
                _add_rests(voice, SLOTS_PER_MEASURE - cursor)

    guitarpro.write(song, output_path)
    return output_path
