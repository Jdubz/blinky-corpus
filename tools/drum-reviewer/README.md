# drum-reviewer

Hand-curate kick/snare ground truth for the 18 `edm/` tracks (#4). It opens a track seeded with the best labels available, shows the audio as waveform, spectrogram and kick/snare band lanes, plays it back with a click at every label, and saves the full corrected list to `edm/drums_human/<stem>.drums.json`. The scorer uses that file once it's marked complete.

The page has no external dependencies and works offline. The server uses only the Python standard library. All audio analysis runs in the browser.

It runs two ways. **Locally** (below) it serves one person from a corpus clone and writes the label file into it. **Hosted** (see [Hosted](#hosted)) it's a public service for many people: edit locks, server-side drafts, and every completed save becomes a pull request that merges by itself once CI passes.

## Run

```sh
python tools/drum-reviewer/server.py                  # http://localhost:8766
python tools/drum-reviewer/server.py --host 0.0.0.0   # serve the LAN (no auth: trusted networks only)
python tools/drum-reviewer/server.py --port 8770 --edm path/to/edm --allow-host studio.example.invalid
```

The page answers only under an IP address, `localhost` or this machine's hostname. For another name, such as a DNS alias, add `--allow-host NAME`. Writes must be same-origin `application/json` POSTs. Together these stop other web pages open in your browser from writing labels, including through DNS rebinding.

Open the page in a desktop Chrome or Firefox. **Help** (or `?`) opens an overlay that explains the page and the review steps; it opens by itself on the first visit. **⚙ Settings** holds every option, each with a plain description: label clicks, volumes, speed, loop, band solo, snapping, reference and hi-hat ticks. Settings are remembered in the browser. The track list (`T`) ranks unfinished tracks worst first, by how much the two best automatic label sets disagree. It also shows each track's reviewed share, counts and whether a draft exists.

## A review pass

1. Open a track. The default view is 6 s. Press `L` to loop the visible range, then `Space`. Label clicks are **off by default**: turn them on in Settings (or `Shift+K` / `Shift+S`) to hear a low click on every kick and a high click on every snare. The toolbar chip shows which are on.
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
| snare | 150 Hz–5 kHz: envelope and onset strength; snare markers; optionally ADTOF hi-hats as faint read-only ticks along the top (Settings, off by default) |
| reviewed | reviewed ranges (green) and unreviewed ones (red) |

The `ref` menu draws ghost ticks along the bottom of each lane from any seed source or reference set in `seeds.json`, for comparison. **ADTOF on drum stem** (`edm/drums_adtof_stem/`: ADTOF run on the HT-Demucs drum stem) finds many of the snares that mix-level ADTOF drops under filters, so a ghost tick with no label above it is worth a listen. Band filtering is zero-phase (forward-backward Butterworth), so the lanes have no filter delay.

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
2. The seed sources in [`seeds.json`](seeds.json), in order, skipping a file marked `skipped`. Today:
   1. `edm/drums_adtof/<stem>.kick_weighted.json`: ADTOF;
   2. `edm/kick_weighted/<stem>.kick_weighted.json`: band-detector labels, known bad. The page shows a red warning.

To add a better seed (for example drums after source separation), put its files in a new directory under `edm/` (kick_weighted schema: `onsets` of `{time, type}`) and add an entry at the top of `seeds.json`. The reviewer reads the corpus's own `seeds.json`, so a hosted service picks the new seed up when it's merged. The page's reference ticks and the label checks cover every source listed.

`seeds.json` can also list `references`: reference-only label sets, in the same entry format. The page offers them as reference ticks, but they never seed a track and the label checks don't compare against them. Today that's `edm/drums_adtof_stem/`. A reference set becomes a seed only by moving its entry into `seeds`, once reviewed ranges show it's better.

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

From the hosted reviewer, a completed save is already a pull request: it merges once CI passes, then step 3 applies. Locally:

1. Commit `edm/drums_human/<stem>.drums.json` on a branch of blinky-corpus and open a PR. CI runs these tests and the scorer's.
2. Once it's merged to `main`, `blinky_scoring.drum_scoring.drum_label_path` picks the human file for that track when `complete` is true. Otherwise it falls back to `drums_adtof/`.
3. A PR in blinky_time (and blinky-firmware) bumps the `corpus` submodule pin. Once that merges, blinky-server's validation suite and the firmware's corpus gate score against the new ground truth.

Partial files can be committed too: the scorer ignores a human file until it's complete.

## Hosted

The hosted reviewer is the same server and page, run as a public service so the owner and helpers can fix labels from anywhere. There's no login: nothing on the page is sensitive, and the protections are below. It runs from the container image in this directory (`Dockerfile`). The page is served by the product site, which sets the API's public URL in the page's `<meta name="drum-reviewer-api">` tag; the server allows that site's origin (`--cors-origin`) and no other.

```sh
docker build -t drum-reviewer tools/drum-reviewer
docker run --read-only --tmpfs /tmp --cap-drop ALL -v state:/state -v corpus:/corpus \
  -e REVIEWER_GITHUB_TOKEN=... drum-reviewer --host 0.0.0.0 --state /state \
  --checkout /corpus/blinky-corpus --git-url https://github.com/Jdubz/blinky-corpus.git \
  --github-repo Jdubz/blinky-corpus --allow-host <api name> --cors-origin <site origin> \
  --trust-cf-connecting-ip
```

- **The corpus** is the service's own clone (`--checkout`), refreshed from `main` every minute. Nothing is ever written into it.
- **Who you are:** an optional name, typed in Settings and remembered in the browser. It's shown with your edit lock and recorded with your saves. Each browser tab also has a random id.
- **Edit locks:** opening a track takes its lock for 15 minutes, renewed while you're active. Everyone else gets the track view-only, with "being edited by …" in the track list. A tab, or an address, holds a limited number of locks. The same name on another machine can take a lock over.
- **Drafts** autosave on the server, per track and per name, so you can carry on from another computer by typing the same name. An anonymous draft belongs to its browser tab. `--import-draft FILE --contributor NAME` copies a local `.drafts/` file into the store (the source is only read).
- **Saving** needs the track 100% reviewed. The server runs the same checks as CI (`labelcheck.py`), then opens a pull request: a branch `labels/<track>-<time>` with one commit of the one file, carrying `Contributed-by:` (the name), `Client-IP-Hash:` (a keyed hash; no address is stored) and `Reviewer-Save-Id:` trailers, with auto-merge (squash) on. The save is refused if anyone saved the track since you opened it (a version check), and while an earlier save of the track is still an open PR. Your work is kept as a draft whatever happens.
- **After the PR:** the service follows it. When `main` moves first, it brings the branch up to date (`main` requires up-to-date branches). Once merged and in the clone, the track is ordinary human ground truth. A PR that's closed unmerged comes back as the contributor's draft.
- **Rate limits** per client address: requests, lock grabs, drafts and saves, plus a global cap on saves (`DEFAULT_LIMITS` in `server.py`). A refusal is HTTP 429 with `Retry-After`.
- **Logs:** one JSON line per request on stdout (`"event": "http"`, with the path, status, time and the client's hash), plus events: `save`, `save_refused`, `save_conflict`, `save_failed`, `lock_acquired`, `rate_limited`, `pr_merged`, `pr_closed`, `refresh_failed`, and others. Every save is also appended to `saves.jsonl` in the state directory.
- **Without a token** (`REVIEWER_GITHUB_TOKEN` empty) locks and drafts work and saving answers 503.

The token is a fine-grained GitHub token on this repository only, with Contents (read and write) and Pull requests (read and write).

### The label checks in CI

Every pull request runs `python tools/drum-reviewer/labelcheck.py --base origin/main` in the required **Scoring tests** job. A PR that touches `edm/drums_human/` must:

1. change only that one file, adding or modifying it (no deletes or renames);
2. match the schema the reviewer writes, against the track's mp3 and manifest duration;
3. be 100% reviewed;
4. be plausible against the file it replaces (the human file on `main`, else the seed it started from): it may not delete more than half of that file's hits (counted regardless of class, so a kick relabelled as a snare isn't a delete), put any class count far outside the range of every reference, or leave a class empty while a reference has hits in it.

The owner's `labels-override` PR label skips step 4 only. A hosted reviewer PR (`labels/` branch) that fails is closed with the reason, and the service returns the labels to the contributor's drafts. On `main`, `--all` checks every human file's schema.

## Tests

- `tests/test_drum_reviewer.py` (pytest, small fixture files) covers listing, seed order, saving, validation, atomic writes, path traversal, the size cap, sha refusal, cross-site and DNS-rebinding refusal, audio range requests, and the scorer reading a saved file.
- `tests/test_drum_reviewer_hosted.py` covers hosted mode against a fake GitHub (`tests/fake_github.py`): locks, versions, drafts that follow a name, rate limits, CORS, the client address, the save → pull request path and its failures, following the PR, importing drafts, seed precedence and the clone's refresh.
- `tests/test_labelcheck.py` covers the CI label checks, including the one-file rule on a scratch git repository.
- `python tools/drum-reviewer/check_js.py` syntax-checks the page's inline scripts with `node --check` and refuses external resources.
- CI's **Hosted reviewer** job builds the image and runs it with the production sandbox (non-root, read-only root filesystem, no capabilities) against the fake GitHub. `tests/e2e_hosted.py` walks the API flow; `tests/browser_hosted.py` drives the page in Chromium from another origin, with two people on one track.
