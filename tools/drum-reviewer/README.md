# drum-reviewer

Hand-curate kick/snare ground truth for the 18 `edm/` tracks (#4). It opens a track seeded with the best labels available, shows the audio as waveform, spectrogram and kick/snare band lanes, plays it back with a click at every label, and saves the full corrected list to `edm/drums_human/<stem>.drums.json`. The scorer uses that file once it's marked complete.

The page has no external dependencies and works offline. The server uses only the Python standard library. All audio analysis runs in the browser.

## Run

```sh
python tools/drum-reviewer/server.py                  # http://localhost:8766
python tools/drum-reviewer/server.py --host 0.0.0.0   # serve the LAN (no auth: trusted networks only)
python tools/drum-reviewer/server.py --port 8770 --edm path/to/edm --allow-host studio.lan
```

The page answers only under an IP address, `localhost` or this machine's hostname. For another name, such as a DNS alias, add `--allow-host NAME`. Writes must be same-origin `application/json` POSTs. Together these stop other web pages open in your browser from writing labels, including through DNS rebinding.

Open the page in a desktop Chrome or Firefox. The track list (`T`) ranks unfinished tracks worst first, by how much ADTOF and the old band labels disagree. It also shows each track's reviewed share, counts and whether a draft exists.

## A review pass

1. Open a track. The default view is 6 s. Press `L` to loop the visible range, then `Space`. The music plays with a low click on every kick and a high click on every snare.
2. Fix what sounds or looks wrong. Hover over a hit and press `K` or `S` to add one, `X` deletes, `C` swaps the class, and `←`/`→` nudge by 1 ms (`Shift` for 5 ms). `N` snaps to the attack. `1`/`2`/`3` slow playback down. `Shift+K`/`Shift+S` mute one click track. `B` solos the low or mid band.
3. `Enter` marks the screen reviewed and moves to the next one (with 10% overlap). Looping playback follows.
4. When the track is 100% reviewed, save with `Ctrl+S`. `U` jumps to any gap you skipped.

Every edit autosaves a draft to `edm/drums_human/.drafts/` (gitignored). Reopening a track with a draft offers to restore it. Save writes the real file and deletes the draft.

## The screen

| lane | shows |
|---|---|
| overview | the whole track, the reviewed share (green), the view box; click to jump |
| spectrogram | 30 Hz–12 kHz, log frequency, 10 ms frames |
| waveform | full-band, down to individual samples; every label is drawn through it |
| kick | 30–150 Hz: envelope (fill) and onset strength (line); kick markers |
| snare | 150 Hz–5 kHz: envelope and onset strength; snare markers; ADTOF hi-hats as faint read-only ticks along the top |
| reviewed | reviewed ranges (green) and unreviewed ones (red) |

The `ref` menu draws ghost ticks along the bottom of each lane from ADTOF or the old band labels, for comparison. Band filtering is zero-phase (forward-backward Butterworth), so the lanes have no filter delay.

**Snap** moves a hit to the nearest strong onset-strength peak within ±25 ms. It's applied when you add a hit (toggle "snap on add") and when you press `N` on a selection. The offset is shown each time.

By default the target is the attack (the 150 Hz–5 kHz onset peak) for both classes. On three corpus tracks, ADTOF kicks sat a median −0.5/+2.5/−0.5 ms from that peak (IQR about ±3 ms). The same kicks sat +3.5 to +5.5 ms from the 30–150 Hz peak, with an IQR around 10 ms: a kick's low-frequency energy builds up after its attack. "own lane band" snaps kicks to the low band instead.

The **offset check** panel gives the median label-to-snap-target offset per class. A consistent non-zero median is a systematic offset, for example a decoder difference between the browser and the labeller, or the kind of late bias in my-spark#1579. Look into it before snapping everything.

**Checks** flags same-class hits closer than 20 ms (`I` jumps to the next one).

## Keyboard

| key | action |
|---|---|
| `Space` | play / pause (from the cursor) |
| `L` | loop the selected region, or the visible range |
| `1` `2` `3` | rate 1.0 / 0.75 / 0.5 (pitch drops with rate; this keeps the clicks sample-accurate) |
| `Shift+K` / `Shift+S` | kick / snare clicks on/off |
| `B` | band solo: full → low → mid |
| `K` / `S` | add kick / snare at the mouse (paused) or the playhead (playing), snapped if on |
| `N` | snap selected hits to the snap target (±25 ms) |
| `C` | change class kick ↔ snare |
| `X` / `Delete` / `Backspace` | delete selected |
| `←` / `→` | nudge selected 1 ms (`Shift`: 5 ms); with nothing selected, pan |
| `Tab` / `Shift+Tab` | select next / previous hit |
| `Ctrl+A` | select hits in the region (or view) |
| `Esc` | clear selection and region |
| `R` / `Shift+R` | mark the region (or view) reviewed / unreviewed |
| `Enter` | mark the view reviewed and go to the next screen |
| `PgDn` / `PgUp` | next / previous screen |
| `U` | next unreviewed gap |
| `I` | next check issue |
| `+` / `-` / `0` | zoom in / out / whole track |
| `Z` | zoom to 1 s around the mouse, and back |
| `Home` / `End` | start / end |
| `Ctrl+Z` / `Ctrl+Shift+Z` (`Ctrl+Y`) | undo / redo |
| `Ctrl+S` | save |
| `T` | track list |

Mouse:

- Click sets the cursor or selects a hit (`Shift` adds to the selection). Dragging a hit moves it.
- Dragging empty space selects a region and the hits in it (only that lane's class when inside a lane).
- Double-click in the kick or snare lane adds a hit.
- The wheel zooms. `Shift`+wheel or a sideways scroll pans.

## Files

The track opens seeded from the first of these that exists:

1. `edm/drums_human/<stem>.drums.json`: existing human ground truth.
2. `edm/drums_adtof/<stem>.kick_weighted.json`: ADTOF, unless marked `skipped`.
3. `edm/kick_weighted/<stem>.kick_weighted.json`: band-detector labels, known bad. The page shows a red warning.

Saved file, `edm/drums_human/<stem>.drums.json`, one onset per line so corrections diff cleanly:

```json
{
  "schema": 1,
  "track": "techno-minimal-01",
  "audio_sha256": "<sha256 of edm/techno-minimal-01.mp3>",
  "duration": 123.6898,
  "started_from": {"source": "drums_adtof", "file": "edm/drums_adtof/techno-minimal-01.kick_weighted.json", "sha256": "<of that file>", "generator": {"algorithm": "adtof_frame_rnn", "...": "..."}},
  "reviewed_ranges": [[0.0, 61.2], [64.0, 123.6898]],
  "complete": false,
  "updated": "2026-11-24T21:03:00+00:00",
  "onsets": [
    {"time": 0.4821, "type": "kick"},
    {"time": 0.9617, "type": "snare"}
  ]
}
```

- `onsets` is the complete list, not a diff: kicks and snares only, times in seconds with 4 decimals, sorted. Hi-hats aren't curated; the scorer takes them from the ADTOF file.
- `complete` is set by the server: true only when `reviewed_ranges` covers the whole track (to within 0.1 s at each end).
- `started_from` records the seed's provenance. Once a human file exists, it keeps its original `started_from`.
- `duration` is the decoded length that the reviewed share is measured against.

The server refuses to save (HTTP 409) when the mp3's sha256 doesn't match the sha the page loaded, or doesn't match an existing human file's `audio_sha256`. Labels made against other audio are never silently carried over. It also validates every onset and range, caps the request at 2 MB and writes atomically (temp file + `fsync` + rename).

## How saved labels reach the scorer

1. Commit `edm/drums_human/<stem>.drums.json` on a branch of blinky-corpus and open a PR. CI runs these tests and the scorer's.
2. Once it's merged to `main`, `blinky_scoring.drum_scoring.drum_label_path` picks the human file for that track when `complete` is true. Otherwise it falls back to `drums_adtof/`.
3. A PR in blinky_time (and blinky-firmware) bumps the `corpus` submodule pin. Once that merges, blinky-server's validation suite and the firmware's corpus gate score against the new ground truth.

Partial files can be committed too: the scorer ignores a human file until it's complete.

## Tests

`tests/test_drum_reviewer.py` (pytest, small fixture files) covers listing, seed order, saving, validation, atomic writes, path traversal, the size cap, sha refusal, cross-site and DNS-rebinding refusal, audio range requests, and the scorer reading a saved file. `python tools/drum-reviewer/check_js.py` syntax-checks the page's inline scripts with `node --check` and refuses external resources. CI runs both.
