# Claude Code Prompt — GuitarSet Download & Basic Pitch Evaluation

Paste this prompt directly into Claude Code CLI to set up the dataset and run the first evaluation.

---

## PROMPT

We are building an AI-powered guitar transcription pipeline that converts high-quality WAV audio into Guitar Pro (.gp5) files with full articulation support (slides, bends, hammer-ons, pull-offs). This is the first evaluation phase using GuitarSet as our benchmark dataset.

Please do the following steps in order, confirming each step before moving to the next:

### Step 1 — Install dependencies
Install all required Python packages:
```
pip install mirdata basic-pitch jams numpy librosa pretty_midi mir_eval pyguitarpro
```
Also check that `demucs` is available and install if not:
```
pip install demucs
```

### Step 2 — Download GuitarSet
Use mirdata to download GuitarSet to a local folder called `./data/guitarset`:
```python
import mirdata
guitarset = mirdata.initialize('guitarset', data_home='./data/guitarset')
guitarset.download()
```
Confirm the download completed and print how many tracks are available.

### Step 3 — Explore the dataset
For the first 3 tracks, print:
- Track ID
- Style (Jazz/Rock/etc)
- Tempo (BPM)
- Mode (solo/comp)
- Audio file paths available (mic, mix, hex)
- Number of annotated notes per string
- Chord annotations summary

### Step 4 — Run Basic Pitch on 5 solo tracks
Select 5 tracks where mode == 'solo' across different styles.
For each track:
1. Run Basic Pitch on the `audio_mic` WAV file
2. Save the output MIDI to `./output/midi/<track_id>.mid`
3. Load the JAMS ground truth annotations
4. Score the MIDI output against the ground truth using mir_eval:
   - Note precision, recall, F1 (onset only, 50ms tolerance)
   - Note precision, recall, F1 (onset + offset, 50ms tolerance)
   - Note precision, recall, F1 (onset + pitch, 50ms tolerance)
5. Print results per track in a clear table

### Step 5 — Summarize results
Print a summary table showing:
- Average F1 scores across all 5 tracks
- Best performing track and why (style, tempo, complexity)
- Worst performing track and why
- Specific failure patterns observed (wrong octave, timing drift, spurious notes, missed notes)
- Recommendation for next parameter or technique to try

### Step 6 — Save a results log
Save all results to `./output/evaluation_log.json` with:
- Timestamp
- Basic Pitch version used
- Per-track scores
- Summary statistics
- Observations and failure patterns
- Suggested next steps

### Notes
- Use `audio_mic` for all tests (closest to real-world input)
- If any track fails, log the error and continue with the rest
- Do not modify the GuitarSet annotations
- All output goes to `./output/` directory
- Print progress as you go so we can follow along
