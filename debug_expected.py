"""
Analysis of what the first 4 bars of 'In My Life' by Kelly Valleau ACTUALLY contain,
based on the CQT analysis from debug_audio.py.

Song is in A major. BPM=106. Song starts at ~1.65s.
Bar duration = 4 * 60/106 = 2.264s

Bar 1: 1.65s - 3.91s
Bar 2: 3.91s - 6.17s (this bar has the D-E3-E3+G#4 figure)
Bar 3: 6.17s - 8.44s (repeat of bar 1)
Bar 4: 8.44s - 10.70s (repeat of bar 2)

Let me map the onsets to what's actually being played:

=== BAR 1 (1.65s - 3.91s) ===
t=1.653: A2 bass (strongest), C#4 melody (strong)  => A2 + C#4
t=2.219: A2 bass (strongest), C#4 melody            => A2 + C#4
t=2.795: A2 bass (strongest), A3 + C#4              => A2 + A3 + C#4 (strum/arpeggio)
t=2.859: Same as above, just continuation/re-attack
t=3.061: C#3 bass (bin 9=C#3), D4 melody (bin 22)   => C#3 + D4
t=3.371: D3 bass (bin 10), E4 melody (bin 24)        => D3 + E4
t=3.648: E3 bass (bin 12), E4 (bin 24) + G#4 (bin 28) => E3 + E4 + G#4

=== BAR 2 (3.91s - 6.17s) ===
(There's a gap from 3.648 to 6.101 - the G#4 sustains through here)
t=6.101: A2 bass, C#4... this is actually the start of bar 3 repeat

Wait, let me recalculate. With BPM=106:
- Beat = 60/106 = 0.566s
- Bar = 4 beats = 2.264s

If the song starts at beat 1 of bar 1 at t=1.653:
Bar 1: 1.653 - 3.917
Bar 2: 3.917 - 6.181
Bar 3: 6.181 - 8.445
Bar 4: 8.445 - 10.709

But looking at the onsets:
- The pattern 1.653-3.648 is the first "phrase"
- Then there's a gap from 3.648 to 6.101
- 6.101-8.171 is the second "phrase" (same pattern)
- Then gap again

So bars 1-2 = one phrase, bars 3-4 = repeat.

Actually, looking more carefully at the beat times from librosa:
beat_times: 1.696, 2.261, 2.827, 3.392, 3.957, 4.523, 5.088, 5.653, 6.208, 6.784...

So beat structure (1-indexed):
Beat 1.1 = 1.696 (~onset at 1.653)
Beat 1.2 = 2.261 (~onset at 2.219)
Beat 1.3 = 2.827 (~onset at 2.795)
Beat 1.4 = 3.392 (~onset at 3.371)
Beat 2.1 = 3.957
Beat 2.2 = 4.523
Beat 2.3 = 5.088
Beat 2.4 = 5.653
Beat 3.1 = 6.208 (~onset at 6.176)
...

So the phrase spans 2 bars:
Bar 1:
  Beat 1 (1.653): A2 + C#4
  Beat 2 (2.219): A2 + C#4
  Beat 3 (2.795): A2 + A3 + C#4
  Beat 3.5 (3.061): C#3 + D4
  Beat 4 (3.371): D3 + E4
Bar 2:
  Beat 1-ish (3.648): E3 + E4 + G#4
  Then G#4 sustains through the rest of bar 2 (no new onsets until bar 3)

This makes sense for a fingerstyle intro! The iconic "In My Life" intro.

Current problems:
1. t=1.653: Detects C#4 only (MISSING A2 bass)
2. t=2.219: Detects C#4 only (MISSING A2 bass)
3. t=2.795: Missing entirely (or merged with 2.859)
4. t=2.859: Detects A2 only (MISSING A3, C#4 melody)
5. t=3.061: Detects C#3 only (MISSING D4 melody)
6. t=3.371: Detects E4 only (MISSING D3 bass)
7. t=3.648: Detects G#4 only (MISSING E3 bass, E4)
8. Bar 2 sustained G#4 is missing entirely

So the bass-harmonic filtering is too aggressive - it's removing REAL notes.
"""
print("See source code for analysis notes")
