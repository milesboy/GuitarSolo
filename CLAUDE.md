# Project: Song → Playable Guitar Version

## What This Is

A tool that takes any song and turns it into something a guitarist can actually play **tonight**.

Not a perfect transcription tool. A *useful* one.

Target user: intermediate guitarist who hears a song, wants to play it, and doesn't want to spend 2 hours on YouTube or pay $5 for a bad tab site PDF.

---

## Core Problem We Solve

Most players don't want note-perfect transcription. They want:
- The chords
- The groove / strumming feel
- The hard spots isolated so they can practice them
- An easier version if the original is too difficult
- Something actionable fast

---

## Roadmap

### 30-Day MVP
- [ ] Upload audio (mp3/wav/YouTube URL)
- [ ] Detect tempo, key, chords
- [ ] Split into sections: intro / verse / chorus / bridge
- [ ] Loop and slow down individual sections
- [ ] Export a simple chord chart (text or basic PDF)

### 60-Day Version
- [ ] Simplified guitar arrangement (stripping complex voicings into playable shapes)
- [ ] Capo suggestions ("Play in G with capo 2 instead of A")
- [ ] Difficulty tiers: Easy / Medium / Original
- [ ] Alternate tuning suggestions where relevant

### 90-Day Version
- [ ] Clean, polished UX
- [ ] Practice history (what you've worked on, how long)
- [ ] PDF export with chord diagrams
- [ ] Demo song library (5–10 examples covering different styles)

---

## Pricing Strategy

Start simple. Early users > pricing perfection.

**Option A:** Free tier (3 songs/month) + $9/mo unlimited  
**Option B:** One-time $19 purchase for beta access  
**Option C:** Pay-per-song ($1–2 per processed song)

Revisit after 50 paying users.

---

## Tech Stack Considerations

- **Audio analysis:** `librosa`, `essentia`, or Spotify's `basic-pitch` for chord/pitch detection
- **Chord detection:** `chord-extractor`, `madmom`, or a fine-tuned model via Replicate
- **Tempo/beat tracking:** `librosa.beat.beat_track`
- **Section segmentation:** structural analysis via `msaf` or manual heuristics
- **Frontend:** React + Tailwind (fast to build, easy to demo)
- **Export:** jsPDF or a server-side PDF renderer
- **Storage:** Supabase or S3 for audio files

---

## What "Good" Looks Like

A user uploads "Wonderwall." Within 60 seconds they have:
1. Key: G major (capo 2, play in A shapes)
2. Tempo: 87 BPM
3. Chords: Em7 – G – Dsus4 – A7sus4 (looped, slowable)
4. Sections labeled
5. A chord chart they can print or screenshot

That's the demo. Build toward that.

---

## What This Is NOT

- Not a full tab/notation editor (that's a different product)
- Not targeting professional musicians or music producers
- Not competing with Sibelius, Guitar Pro, or Soundslice directly
- Not trying to be "AI for music" broadly — that's too vague to ship or sell

---

## Why This Works Commercially

- Easy to explain in one sentence
- Easy to demo live (upload → output in under a minute)
- Easy to price (solve a specific problem, charge for access)
- The market is huge (200M+ guitarists globally, millions learning every year)
- SEO goldmine: every song title + "guitar chords" is a search query

---

## Open Questions

- Do we handle vocals / melody line or just chords?
- How do we handle copyright on chord charts? (Facts aren't copyrightable; arrangements may be)
- Mobile-first or desktop-first UX?
- Do we let users save/edit the chord charts manually?

---

## Notes for Claude

- When writing code for this project, prioritize **working over perfect**
- Prefer libraries with good Python/JS ecosystems over rolling custom ML
- The demo experience is the product — optimize for first-run wow factor
- Keep the UI simple: guitarists are not always tech-savvy
- When in doubt, cut scope and ship the simpler version
