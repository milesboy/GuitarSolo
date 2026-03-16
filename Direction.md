# CLAUDE.md — Guitar Transcription Pipeline

This file gives Claude Code full context on what we are building, why, and how.
Read this at the start of every session before taking any action.

---

## Project Goal

Build a **web-based AI guitar transcription product** that converts high-quality audio
(WAV/FLAC) into accurate Guitar Pro (.gp5) files, including guitar-specific articulations
such as slides, bends, hammer-ons, and pull-offs. No existing product does this well.
This is our core competitive advantage.

---

## What We Are Building

A server-side transcription pipeline exposed as a web product. Users upload audio,
receive a Guitar Pro file back. The heavy processing runs on our servers so the core
IP is never exposed to users or competitors.

### Target Users
Guitarists who want to learn songs by ear but need accurate tabs — especially
articulations that current tools (Klangio, AnthemScore, Tabtify) miss entirely.

### Key Differentiators vs Competitors
| Feature | Competitors | Us |
|---|---|---|
| Guitar Pro output | Some | ✅ |
| Slides / bends / hammer-ons | ❌ None | ✅ Goal |
| Iterative AI refinement loop | ❌ None | ✅ Core feature |
| Scores against ground truth | ❌ None | ✅ Core feature |
| One-shot only | ✅ All of them | ❌ We iterate |

---

## Technical Architecture

### Transcription Pipeline (server-side)
```
Input WAV/FLAC (96kHz/24-bit preferred)
    ↓
Demucs          — isolate guitar stem from mix
    ↓
Basic Pitch     — primary note/pitch detection (Spotify, open source)
    ↓
Transition      — classify note transitions:
Analyzer            jump + onset   = picked note
                    jump no onset  = hammer-on / pull-off
                    smooth glide   = slide
                    pitch rise     = bend
    ↓
pyguitarpro     — write .gp5 with full articulation tags
    ↓
Scoring Loop    — render MIDI via fluidsynth, compare to original
    ↓
Claude API      — AI reasoning loop: analyze failures, suggest next technique
    ↓
Output .gp5
```

### AI Refinement Loop
Claude is used as the reasoning engine inside the feedback loop. It does NOT
brute-force parameters — it analyzes scoring output and decides which technique
or parameter to try next. This is the novel part of the architecture.

### Scoring Metrics
- Chroma similarity (right pitches at right times)
- Onset alignment (note attack timing)
- Note histogram match (distribution of notes vs original)
- mir_eval note precision/recall/F1 at 50ms tolerance

---

## Development Phases

### Phase 1 — GuitarSet Baseline (CURRENT)
- Dataset: GuitarSet (360 excerpts, acoustic guitar, JAMS annotations)
- Goal: Establish baseline F1 score with Basic Pitch on `audio_mic` tracks
- Ground truth: JAMS per-string note annotations
- Success metric: F1 > 0.80 on solo tracks before moving to Phase 2
- Output: `./output/evaluation_log.json` per run

### Phase 2 — GOAT Dataset + Articulations
- Dataset: GOAT (5.9hrs, electric guitar, actual Guitar Pro annotations)
- Goal: Add articulation detection (slides, bends, hammer-ons)
- Ground truth: Guitar Pro files paired with DI audio
- New component: Transition Analyzer

### Phase 3 — GAPS Dataset + Generalization
- Dataset: GAPS (14hrs, 200+ performers, classical guitar)
- Goal: Generalization across timbres and recording conditions
- Test: Zero-shot performance on unseen recordings

### Phase 4 — Real-world Validation
- Sources: Ultimate Guitar .gp5 tabs + Qobuz/HDTracks hi-res WAV
- Goal: Validate on commercial recordings
- Legal note: Use only for development, not model training at scale
- Songs to start with: Beatles Blackbird, Julia; Chet Atkins; Tommy Emmanuel

---

## Technology Stack

### Audio Processing
- `demucs` — stem separation (Meta, state of the art)
- `basic-pitch` — pitch/note detection (Spotify)
- `librosa` — audio analysis and feature extraction
- `aubio` — CLI tempo/onset detection (supplementary)
- `fluidsynth` — MIDI to audio rendering for scoring loop

### Transcription & Notation
- `pyguitarpro` — read/write Guitar Pro (.gp3/.gp4/.gp5) files
- `pretty_midi` — MIDI manipulation
- `mirdata` — dataset loading (GuitarSet, etc.)
- `mir_eval` — standard music transcription evaluation metrics
- `jams` — JAMS annotation format parsing

### AI Loop
- Anthropic Claude API (`claude-sonnet-4-20250514`) — reasoning engine
- Loop: score → analyze failures → suggest next technique → apply → re-score

### Web Product (future)
- Server-side processing only — pipeline never runs client-side
- Credit-based pricing model
- Target: $3-5 per transcription or $15-30/month subscription

---

## File Structure
```
./
├── CLAUDE.md                    ← This file
├── data/
│   └── guitarset/               ← GuitarSet dataset (downloaded)
├── output/
│   ├── midi/                    ← Transcribed MIDI files
│   ├── gp5/                     ← Output Guitar Pro files
│   └── evaluation_log.json      ← Scoring results per run
├── pipeline/
│   ├── transcribe.py            ← Main transcription pipeline
│   ├── score.py                 ← Scoring and comparison logic
│   ├── articulations.py         ← Transition/articulation detector
│   ├── ai_loop.py               ← Claude API refinement loop
│   └── export_gp5.py            ← pyguitarpro export
└── prompts/
    └── guitarset_setup_prompt.md ← Phase 1 CLI prompt
```

---

## Key Decisions Made

1. **Web product, not CLI tool** — protects IP, enables recurring revenue
2. **Basic Pitch over aubio** — better accuracy, actively maintained by Spotify
3. **GuitarSet first** — clean benchmark, comparable to published research
4. **JAMS over GP5 for Phase 1** — richer ground truth than tabs
5. **pyguitarpro for output** — only Python library that writes real .gp5 files
6. **Claude as AI loop** — reasoning over failures, not brute-force parameter search
7. **Credit-based pricing** — lower friction than subscription for initial validation

---

## What Not To Do

- Do NOT use aubio as the primary transcription engine (too inaccurate)
- Do NOT output MIDI only — Guitar Pro is the goal
- Do NOT brute-force all aubio parameters — use Claude to reason about next steps
- Do NOT train on copyrighted commercial recordings at scale without legal clearance
- Do NOT run the transcription pipeline client-side
- Do NOT use MT3 (abandoned, weaker on guitar than Basic Pitch)
- Do NOT use Omnizart as primary transcription (stale since 2021, heavier than Basic Pitch)

---

## Competitive Landscape

| Product | GP5 output | Articulations | Iterative | Status |
|---|---|---|---|---|
| Klangio/Guitar2Tabs | ✅ | ❌ | ❌ | Active, Series A startup |
| AnthemScore | Partial | ❌ | ❌ | Active |
| Tabtify | ✅ | ❌ | ❌ | Small |
| GuitarConvert | ❌ PDF only | ❌ | ❌ | Active |
| MT3 (Google) | ❌ MIDI only | ❌ | ❌ | Abandoned research |
| **This product** | ✅ | ✅ Goal | ✅ | In development |

---

## Session Notes

Add dated notes here as the project evolves.

### Session 1
- Established pipeline direction: Demucs → Basic Pitch → Transition Analyzer → pyguitarpro
- Chose GuitarSet as Phase 1 benchmark
- Decided on web product architecture
- Key insight: articulation detection is the unsolved problem and our moat
- Key insight: AI loop reasoning over failures is novel — no competitor does this
